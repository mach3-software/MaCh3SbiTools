import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from mach3sbitools.data_processors import CompressorBase
from mach3sbitools.simulator import Prior
from mach3sbitools.utils import (
    CAN_DROP_CACHE,
    advise_sequential,
    cgroup_memory_limit,
    drop_from_cache,
    get_logger,
)

#: Written by mach3sbitools.apps.merge_shards. Kept as a literal rather than
#: imported to avoid data_loaders -> apps coupling.
_METADATA_FILENAME = "merge_metadata.json"

#: Two index runs closer than this are served by one read that spans the gap.
#: Reading a little dead data costs far less than a second round trip to a
#: network filesystem.
_COALESCE_GAP_BYTES = 256 * 1024


class TrainingDataset(Dataset):
    def __init__(self, theta_path: Path, x_path: Path, prior: Prior):
        self.theta_path = theta_path
        self.x_path = x_path

        # Read the .npy headers once to learn shape/dtype and where each
        # file's data begins. The memmaps are released immediately: reads go
        # through pread() instead.
        #
        # mmap is the wrong tool for this access pattern. Walking a
        # multi-hundred-GB file maps every page it touches, and under a cgroup
        # memory limit those pages cannot all be resident -- so the kernel
        # faults them in 4 KiB at a time and immediately evicts them again,
        # burning the job's whole memory allowance on reclaim churn. pread
        # turns each batch into one large sequential read and leaves the
        # process's memory footprint flat.
        self._theta_meta = self._read_npy_header(theta_path)
        self._x_meta = self._read_npy_header(x_path)
        self._len = self._theta_meta["shape"][0]

        if self._x_meta["shape"][0] != self._len:
            raise ValueError(
                f"theta has {self._len} rows but x has "
                f"{self._x_meta['shape'][0]}; these files do not match."
            )

        # Opened lazily per worker process, so no descriptor is pickled when
        # the DataLoader spawns workers.
        self._theta_fd: int | None = None
        self._x_fd: int | None = None
        self._fd_pid: int | None = None
        self._drop_cache = self._should_drop_cache()
        self._theta_buf: np.ndarray | None = None
        self._x_buf: np.ndarray | None = None

        self._nuisance_filter = prior.nuisance_filter.cpu().bool()
        self._n_full = int(self._nuisance_filter.numel())
        self._n_kept = int(self._nuisance_filter.sum())

        # Applied to every batch after reading; see set_compressors.
        self._theta_compressor: CompressorBase | None = None
        self._x_compressor: CompressorBase | None = None

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

    def set_compressors(
        self,
        theta_compressor: CompressorBase | None = None,
        x_compressor: CompressorBase | None = None,
    ) -> None:
        """
        Attach fitted compressors, applied to every sample after reading.

        Doing it here rather than at the call sites is what keeps the whole
        pipeline consistent: the density estimator is sized from this same
        dataset, so the network, the training batches and the validation
        batches all see the compressed dimensionality automatically.

        Note this does not reduce I/O -- full-width rows are still read off
        disk and then projected. To save reads the projection has to be
        applied to the stored data instead.

        :param theta_compressor: Fitted compressor for theta, or ``None``.
        :param x_compressor: Fitted compressor for x, or ``None``.
        """
        self._theta_compressor = theta_compressor
        self._x_compressor = x_compressor

    @property
    def has_compressors(self) -> bool:
        """True once :meth:`set_compressors` has attached at least one."""
        return self._theta_compressor is not None or self._x_compressor is not None

    def _compress(
        self, theta: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply whichever compressors are attached. No-op when there are none."""
        if self._theta_compressor is not None:
            theta = self._theta_compressor.transform(theta)
        if self._x_compressor is not None:
            x = self._x_compressor.transform(x)
        return theta, x

    @staticmethod
    def _read_npy_header(path: Path) -> dict:
        """Grab shape, dtype and data offset without holding the file open."""
        mm = np.load(path, mmap_mode="r")
        meta = {
            "shape": tuple(mm.shape),
            "dtype": mm.dtype,
            "offset": int(mm.offset),
            "row_bytes": int(mm.shape[1]) * mm.dtype.itemsize,
        }
        del mm
        return meta

    def _should_drop_cache(self) -> bool:
        """
        Decide whether to discard page cache as we read.

        Caching only pays off if the data can actually stay resident between
        epochs. When the files dwarf the job's memory limit they cannot, and
        leaving the cache to fill just drives the kernel into continuous
        reclaim -- the pathology where mapped/cached memory climbs to the
        cgroup ceiling, collapses, and climbs again while throughput craters.
        In that regime dropping each range after use keeps the footprint flat
        and costs nothing, because those pages were never going to be reused.

        Set ``MACH3SBI_DROP_PAGE_CACHE`` to 0 or 1 to override.
        """
        override = os.environ.get("MACH3SBI_DROP_PAGE_CACHE")
        if override is not None:
            return override.strip() not in ("0", "false", "False", "")

        if not CAN_DROP_CACHE:
            return False

        limit = cgroup_memory_limit()
        if limit is None:
            return False  # no limit to blow; let the kernel cache freely

        total = sum(
            m["shape"][0] * m["row_bytes"] for m in (self._theta_meta, self._x_meta)
        )
        # Half the limit, since the cache competes with the prefetch queue,
        # pinned buffers and the process itself.
        drop = bool(total > 0.5 * limit)
        if drop:
            get_logger().info(
                "Dataset is %.0f GB against a %.0f GB cgroup limit; dropping page "
                "cache behind reads to avoid reclaim thrashing",
                total / 1e9,
                limit / 1e9,
            )
        return drop

    def _ensure_open(self) -> None:
        """
        Open both files in this process, reopening after a fork.

        DataLoader workers inherit the parent's descriptors on fork. pread is
        stateless so sharing would technically work, but each process gets its
        own here to keep readahead state independent.
        """
        pid = os.getpid()
        if self._theta_fd is not None and self._fd_pid == pid:
            return

        self._theta_fd = os.open(self.theta_path, os.O_RDONLY)
        self._x_fd = os.open(self.x_path, os.O_RDONLY)
        self._fd_pid = pid

        advise_sequential(self._theta_fd)
        advise_sequential(self._x_fd)

    @staticmethod
    def _contiguous_runs(idx: np.ndarray, row_bytes: int) -> list[tuple[int, int]]:
        """
        Split ascending indices into (start_row, n_rows) spans to read.

        Runs separated by a gap smaller than :data:`_COALESCE_GAP_BYTES` are
        merged into one read that spans the gap; the unwanted rows are
        discarded afterwards. This keeps a strided access pattern -- which is
        what ``DistributedSampler`` produces under DDP -- from turning into
        one syscall per row.
        """
        if idx.size == 0:
            return []

        breaks = np.flatnonzero(np.diff(idx) != 1)
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks + 1, [idx.size]))

        runs: list[tuple[int, int]] = []
        for s_i, e_i in zip(starts, ends):
            start, n = int(idx[s_i]), int(e_i - s_i)
            if runs:
                prev_start, prev_n = runs[-1]
                gap = start - (prev_start + prev_n)
                if 0 <= gap * row_bytes <= _COALESCE_GAP_BYTES:
                    runs[-1] = (prev_start, start + n - prev_start)
                    continue
            runs.append((start, n))
        return runs

    def _pread_rows(
        self, fd: int, meta: dict, idx: np.ndarray, buf_attr: str
    ) -> np.ndarray:
        """Read the rows named by *idx* (ascending) into a reused buffer."""
        n_cols = meta["shape"][1]
        row_bytes = meta["row_bytes"]
        runs = self._contiguous_runs(idx, row_bytes)

        span_rows = sum(n for _, n in runs)
        buf = getattr(self, buf_attr)
        if buf is None or buf.shape[0] < span_rows or buf.shape[1] != n_cols:
            buf = np.empty((span_rows, n_cols), dtype=meta["dtype"])
            setattr(self, buf_attr, buf)

        raw = buf.reshape(-1).view(np.uint8)
        rows_of: list[np.ndarray] = []
        filled = 0

        for start, n in runs:
            nbytes = n * row_bytes
            target = raw[filled * row_bytes : filled * row_bytes + nbytes]
            file_offset = meta["offset"] + start * row_bytes
            self._pread_into(fd, target, file_offset)
            if self._drop_cache:
                drop_from_cache(fd, file_offset, nbytes)
            rows_of.append(np.arange(start, start + n))
            filled += n

        # Coalesced reads may have pulled in rows nobody asked for; select the
        # requested ones back out of the buffer.
        available = np.concatenate(rows_of) if rows_of else np.empty(0, dtype=int)
        if available.size == idx.size and np.array_equal(available, idx):
            return np.asarray(buf[:filled])

        lookup = {int(r): i for i, r in enumerate(available)}
        take = np.fromiter((lookup[int(i)] for i in idx), dtype=np.intp, count=idx.size)
        return np.asarray(buf[:filled][take])

    @staticmethod
    def _pread_into(fd: int, target: np.ndarray, offset: int) -> None:
        """Fill *target* from *fd* at *offset*, looping until it is full."""
        want = target.nbytes
        done = 0
        while done < want:
            chunk = os.pread(fd, want - done, offset + done)
            if not chunk:
                raise OSError(
                    f"Unexpected end of file at offset {offset + done} "
                    f"(wanted {want} bytes, got {done})"
                )
            target[done : done + len(chunk)] = np.frombuffer(chunk, dtype=np.uint8)
            done += len(chunk)

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

    def _read_rows(self, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """
        Read a block of rows from both files.

        Indices are sorted before reading so the access pattern runs forward
        through each file, then the caller's ordering is restored with an
        in-RAM gather.
        """
        self._ensure_open()
        assert self._theta_fd is not None and self._x_fd is not None

        idx = np.asarray(indices, dtype=np.int64)

        order = np.argsort(idx, kind="stable")
        sorted_idx = idx[order]

        theta = self._pread_rows(
            self._theta_fd, self._theta_meta, sorted_idx, "_theta_buf"
        )
        x = self._pread_rows(self._x_fd, self._x_meta, sorted_idx, "_x_buf")

        if not np.array_equal(order, np.arange(order.size)):
            inverse = np.empty_like(order)
            inverse[order] = np.arange(order.size)
            theta = theta[inverse]
            x = x[inverse]

        return theta, x

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

        # .copy() because _read_rows may hand back a view of the reused
        # buffer, which the next batch would overwrite underneath us.
        theta_batch = self._filter_theta(torch.from_numpy(theta_np.copy()).float())
        x_batch = torch.from_numpy(x_np.copy()).float()

        return self._compress(theta_batch, x_batch)

    def __getitem__(self, idx):
        """Single row, or a slice. Both route through the pread path."""
        if isinstance(idx, slice):
            indices = list(range(*idx.indices(self._len)))
            theta_np, x_np = self._read_rows(indices)
            theta = self._filter_theta(torch.from_numpy(theta_np.copy()).float())
            x = torch.from_numpy(x_np.copy()).float()
            return self._compress(theta, x)

        theta_np, x_np = self._read_rows([int(idx)])
        theta = self._filter_theta(torch.from_numpy(theta_np[0].copy()).float())
        x = torch.from_numpy(x_np[0].copy()).float()
        return self._compress(theta, x)
