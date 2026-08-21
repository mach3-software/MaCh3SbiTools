"""
Dataset implementation for feather-based simulation files.
"""

import bisect
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm

from mach3sbitools.simulator import Prior
from mach3sbitools.utils import FeatherFileHandle, peek_num_rows


class TrainingDataset(Dataset):
    """
    Row-level PyTorch dataset over a folder of ``.feather`` simulation files.

    Each file is memory-mapped and opened lazily on first access; rows are
    indexed globally across all files, so this can be passed straight into
    a ``DataLoader`` without any pre-loading step. Each ``DataLoader``
    worker opens its own file handles (see ``__getstate__``/``__setstate__``).

    Files must have been written with ``compression="uncompressed"``
    (see :func:`mach3sbitools.utils.to_feather`).
    """

    def __init__(self, data_folder: Path, prior: Prior):
        """
        :param data_folder: Directory containing ``.feather`` files.
        :param prior: Prior providing ``nuisance_filter`` used to select
            which parameters are kept in *theta* on load.
        :raises FileNotFoundError: If *data_folder* contains no ``.feather``
            files.
        """
        if not isinstance(data_folder, Path):
            data_folder = Path(data_folder)

        self.data_folder = data_folder
        self.files = sorted(data_folder.glob("*.feather"))
        if not self.files:
            raise FileNotFoundError(f"No .feather files found in {data_folder}")

        self.prior = prior
        self.lengths = [peek_num_rows(f) for f in self.files]
        self.cumsum = np.cumsum([0, *self.lengths])
        self._handles: dict[int, FeatherFileHandle] = {}

    def __len__(self) -> int:
        """Total number of rows across all feather files."""
        return int(self.cumsum[-1])

    def _handle(self, file_idx: int) -> FeatherFileHandle:
        handle = self._handles.get(file_idx)
        if handle is None:
            handle = FeatherFileHandle(self.files[file_idx])
            self._handles[file_idx] = handle
        return handle

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the ``(theta, x)`` pair at global row index *idx*.

        :param idx: Global row index across all files.
        :returns: Tuple of ``(theta, x)`` float tensors for a single row.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        file_idx = bisect.bisect_right(self.cumsum, idx) - 1
        local_idx = idx - self.cumsum[file_idx]
        handle = self._handle(file_idx)

        theta = handle.theta[local_idx]
        x = handle.x[local_idx]

        if self.prior.nuisance_filter is not None:
            nuisance_filter = self.prior.nuisance_filter
            if isinstance(nuisance_filter, torch.Tensor):
                nuisance_filter = nuisance_filter.to("cpu").numpy()
            theta = theta[nuisance_filter]

        theta_t = torch.from_numpy(theta.astype(np.float32, copy=True))
        x_t = torch.from_numpy(x.astype(np.float32, copy=True))
        return theta_t, x_t

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def close(self) -> None:
        """Close any open memory-mapped file handles in this process."""
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def to_tensor_dataset(
        self, device: str = "cpu", verbose: bool = True
    ) -> TensorDataset:
        """
        Materialize every row into a flat :class:`~torch.utils.data.TensorDataset`
        in RAM. Kept for backward compatibility; prefer passing this dataset
        directly to a ``DataLoader`` instead, since this defeats lazy loading.

        :param device: Target device for the output tensors.
        :returns: A :class:`~torch.utils.data.TensorDataset` of
            ``(theta_tensor, x_tensor)``.
        """
        all_theta, all_x = [], []

        for idx in tqdm(
            range(len(self)), desc="Pre-loading dataset", disable=not verbose
        ):
            theta, x = self[idx]
            all_theta.append(theta)
            all_x.append(x)

        theta_tensor = torch.stack(all_theta, dim=0).to(device)
        x_tensor = torch.stack(all_x, dim=0).to(device)

        if verbose:
            print(
                f"Loaded {theta_tensor.shape[0]:,} simulations | "
                f"θ: {theta_tensor.shape[1]}D  x: {x_tensor.shape[1]}D | "
                f"RAM: {(theta_tensor.nbytes + x_tensor.nbytes) / 1e9:.2f} GB"
            )

        return TensorDataset(theta_tensor, x_tensor)
