"""
Memory-mapped ``(theta, x)`` dataset backed by a pair of ``.npy`` files.

The files are produced by :func:`~mach3sbitools.apps.merge_shards.merge_shards_module`
and are opened with ``mmap_mode="r"`` so that only the rows actually touched
are paged in.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from mach3sbitools.simulator import Prior

Index = int | slice | np.ndarray


class TrainingDataset(Dataset):
    """
    Lazily memory-mapped dataset over ``theta.npy`` / ``x.npy``.

    Nuisance parameters are dropped from *theta* on read, so the dataset
    always yields the active parameter set described by
    :attr:`~mach3sbitools.simulator.Prior.nuisance_filter`.

    The memmaps themselves are opened on first access rather than in
    :meth:`__init__`, so nothing large is pickled when a ``DataLoader``
    spawns worker processes.
    """

    def __init__(self, theta_path: Path, x_path: Path, prior: Prior) -> None:
        """
        Open the dataset headers and validate them against *prior*.

        :param theta_path: Path to the ``theta.npy`` memmap file.
        :param x_path: Path to the ``x.npy`` memmap file.
        :param prior: Prior supplying the nuisance keep-mask.
        :raises ValueError: If the two files disagree on row count, or if
            *theta*'s width does not match the prior's parameter count.
        """
        self.theta_path = Path(theta_path)
        self.x_path = Path(x_path)

        # Read the headers only — np.load with mmap_mode does not read data.
        theta_shape = np.load(self.theta_path, mmap_mode="r").shape
        x_shape = np.load(self.x_path, mmap_mode="r").shape

        self._nuisance_filter = prior.nuisance_filter.cpu().numpy()

        if theta_shape[0] != x_shape[0]:
            raise ValueError(
                f"theta and x disagree on row count: {theta_shape[0]} vs {x_shape[0]}"
            )

        if theta_shape[-1] != len(self._nuisance_filter):
            raise ValueError(
                f"theta has {theta_shape[-1]} parameters but the prior describes "
                f"{len(self._nuisance_filter)}"
            )

        self._len = int(theta_shape[0])
        self._x_dim = int(x_shape[-1])

        self._theta: np.ndarray | None = None
        self._x: np.ndarray | None = None

    @property
    def theta_dim(self) -> int:
        """
        :returns: Number of active parameters returned by this dataset.
        """
        return int(self._nuisance_filter.sum())

    @property
    def x_dim(self) -> int:
        """
        :returns: Width of the observable vector.
        """
        return self._x_dim

    def _ensure_open(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Open the memmaps if this process has not done so yet.

        :returns: The ``(theta, x)`` memmaps.
        """
        if self._theta is None or self._x is None:
            self._theta = np.load(self.theta_path, mmap_mode="r")
            self._x = np.load(self.x_path, mmap_mode="r")
        return self._theta, self._x

    def __len__(self) -> int:
        """
        :returns: Number of rows in the dataset.
        """
        return self._len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch one row, a slice of rows, or an array of row indices.

        The nuisance filter is always applied to the final (parameter) axis,
        so the same call works for scalar and batched indices alike.

        :param index: Row index, slice, or integer array.
        :returns: Tuple of ``(theta, x)`` float32 tensors.
        """
        theta_map, x_map = self._ensure_open()

        # np.array copies: a memmap slice is read-only, and torch refuses to
        # wrap a non-writable buffer without warning.
        theta = torch.from_numpy(np.array(theta_map[index], dtype=np.float32))
        x = torch.from_numpy(np.array(x_map[index], dtype=np.float32))

        return theta[..., self._nuisance_filter], x

    def __getitems__(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Batched fetch used by ``DataLoader`` to avoid per-row overhead.

        Returns the batch already stacked rather than a list of rows. The
        conventional list return would be unbound into one tensor per row and
        then restacked by the collate function into exactly the tensor read
        here — for a 2048-row batch that round trip costs several times the
        read itself. :func:`collate_batch` passes this straight through.

        Shuffled training indices arrive in random order, which makes the
        memmap read jump around the file. Sorting them first turns the read
        into a forward scan over the pages the batch needs. A batch is an
        unordered set as far as the loss is concerned, so this changes only
        the read pattern.

        :param indices: Row indices to fetch.
        :returns: Tuple of stacked ``(theta, x)`` tensors.
        """
        return self[np.sort(np.asarray(indices))]
