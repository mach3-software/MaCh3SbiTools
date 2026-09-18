import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from mach3sbitools.simulator import Prior
from mach3sbitools.utils import get_logger

#: Written by mach3sbitools.apps.merge_shards. Kept as a literal rather than
#: imported to avoid data_loaders -> apps coupling.
_METADATA_FILENAME = "merge_metadata.json"


class TrainingDataset(Dataset):
    def __init__(self, theta_path: Path, x_path: Path, prior: Prior):
        self.theta_path = theta_path
        self.x_path = x_path

        # Read only the header here to get shape/dtype/length cheaply,
        # without holding an open memmap across a potential fork.
        theta_header = np.load(theta_path, mmap_mode="r")
        self._len = theta_header.shape[0]
        del theta_header

        # Actual memmaps are opened lazily per worker process (see below),
        # so nothing large is pickled when DataLoader spawns workers.
        # Annotated because the initial None would otherwise be the only type
        # a checker sees for these.
        self._theta: np.ndarray | None = None
        self._x: np.ndarray | None = None

        self._nuisance_filter = prior.nuisance_filter.cpu().bool()
        self._n_full = int(self._nuisance_filter.numel())
        self._n_kept = int(self._nuisance_filter.sum())

        self._check_merge_metadata(prior)

    def _check_merge_metadata(self, prior: Prior) -> None:
        """
        If the data was nuisance-filtered at merge time, verify it was
        filtered with *this* prior.

        The filter is baked into the file, so a mismatched prior would
        otherwise train silently on the wrong parameters -- the column count
        can match by coincidence even when the parameters differ.
        """
        metadata_path = self.theta_path.parent / _METADATA_FILENAME
        if not metadata_path.is_file():
            return

        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            get_logger().warning(f"Could not read {metadata_path}: {exc}")
            return

        if not metadata.get("theta_filtered"):
            return

        merged_names = metadata.get("kept_parameter_names")
        if merged_names is None:
            return

        prior_names = list(prior.prior_data.parameter_names)
        if list(merged_names) != prior_names:
            raise ValueError(
                f"{self.theta_path} was nuisance-filtered at merge time using a "
                f"different prior.\n"
                f"  merged with: {len(merged_names)} params, "
                f"prior expects: {len(prior_names)} params\n"
                f"  first mismatch: "
                f"{next((f'{a!r} != {b!r}' for a, b in zip(merged_names, prior_names) if a != b), 'length differs')}\n"
                f"Re-merge with this prior, or point at the matching prior."
            )

        get_logger().info(
            f"Data pre-filtered at merge time: {len(merged_names)} theta columns "
            f"(matches prior)"
        )

    def _ensure_open(self):
        if self._theta is None:
            self._theta = np.load(self.theta_path, mmap_mode="r")
            self._x = np.load(self.x_path, mmap_mode="r")

    def _filter_theta(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Apply the nuisance filter along the last axis.

        Handles both layouts: theta straight from an unfiltered merge (full
        width, mask it here) and theta that was already filtered at merge
        time (correct width, nothing to do). Branching on the trailing
        dimension rather than ``shape[0]`` keeps this correct for single
        rows, batches, and the batch-of-batches case alike.
        """
        width = theta.shape[-1]

        if width == self._n_kept:
            return theta  # already filtered on disk
        if width == self._n_full:
            return theta[..., self._nuisance_filter]

        raise ValueError(
            f"theta has {width} columns along its last axis; expected "
            f"{self._n_full} (unfiltered) or {self._n_kept} (pre-filtered at "
            f"merge time). Check that {self.theta_path} matches this prior."
        )

    def __len__(self):
        return self._len

    @staticmethod
    def _as_slice(idx: np.ndarray) -> slice | None:
        """
        Express *idx* as a constant-stride slice, or ``None`` if it isn't one.

        Both samplers we actually run hit this path. ``shuffle=False`` on a
        single rank gives contiguous runs (stride 1); under DDP,
        ``DistributedSampler`` hands rank *r* an interleaved stride of
        ``world_size``. Slicing a memmap lets the kernel see a predictable
        access pattern and read ahead, where fancy indexing degenerates into
        an unordered gather of individual pages.
        """
        if idx.size == 0:
            return None
        if idx.size == 1:
            return slice(int(idx[0]), int(idx[0]) + 1)

        step = int(idx[1]) - int(idx[0])
        if step <= 0:
            return None
        if not np.array_equal(np.diff(idx), np.full(idx.size - 1, step)):
            return None

        return slice(int(idx[0]), int(idx[-1]) + step, step)

    def _read_rows(self, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Pull a block of rows out of the memmaps, copied into worker memory."""
        self._ensure_open()
        assert self._theta is not None and self._x is not None

        idx = np.asarray(indices)

        sel = self._as_slice(idx)
        if sel is not None:
            # Slicing a memmap returns a view; copy here so the page faults
            # happen in this worker rather than later in the main process's
            # pin_memory thread, where they would not overlap with compute.
            return np.array(self._theta[sel]), np.array(self._x[sel])

        # Arbitrary order: fault pages in ascending order, then restore the
        # caller's ordering with an in-RAM gather. Fancy indexing already
        # returns a copy, so no extra np.array() here.
        order = np.argsort(idx)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size)
        sorted_idx = idx[order]

        return self._theta[sorted_idx][inverse], self._x[sorted_idx][inverse]

    def __getitems__(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch a whole batch at once.

        Returns the batch *already collated* as ``(theta, x)``. Building a
        list of per-row tuples here would force the DataLoader's collate to
        re-stack them one row at a time -- at large batch sizes that is
        millions of Python-level tensor allocations per batch, which is far
        more expensive than the reads themselves. ``SBIDataModule`` pairs
        this with a pass-through ``collate_fn``.
        """
        theta_np, x_np = self._read_rows(indices)

        theta_batch = self._filter_theta(torch.from_numpy(theta_np).float())
        x_batch = torch.from_numpy(x_np).float()

        return theta_batch, x_batch

    def __getitem__(self, idx):
        self._ensure_open()

        assert self._theta is not None
        assert self._x is not None

        theta = torch.from_numpy(np.array(self._theta[idx])).float()
        x = torch.from_numpy(np.array(self._x[idx])).float()

        theta = self._filter_theta(theta)

        return theta, x
