"""
PyTorch Lightning data module for SBI simulation datasets.

Dataset sharing strategy
------------------------
This module accepts any map-style ``Dataset`` — including a memory-mapped,
lazily-loading dataset such as
:class:`~mach3sbitools.data_loaders.TrainingDataset` — rather than requiring a
pre-loaded, fully in-RAM ``TensorDataset``.

* Under DDP, Lightning's built-in ``DistributedSampler`` (activated
  automatically when ``strategy="ddp"``) gives each rank a disjoint slice of
  indices, so every GPU only touches its own share of rows.
* When the dataset is backed by memory-mapped ``.npy`` files, ranks on the
  same node share the OS page cache: identical pages aren't duplicated in
  physical RAM even though each rank/worker opens its own ``mmap()``. Across
  nodes there's no such sharing, but each node still only pages in what its
  own ranks actually touch.
* If a plain in-RAM ``TensorDataset`` is passed instead (e.g. for a small
  dataset that comfortably fits in memory), the same code path works
  unchanged — the split and ``DistributedSampler`` only need index-level
  slicing of a map-style ``Dataset``.

``num_workers`` should generally be > 0 when the dataset performs lazy
per-row I/O, so that disk/page-cache reads for the next batch overlap with
GPU compute on the current one. This is the opposite of the old advice for a
fully RAM-resident ``TensorDataset``, where extra worker processes only added
IPC overhead for no benefit.
"""

from __future__ import annotations

import warnings
from collections.abc import Sized
from typing import cast

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, Subset, default_collate

from mach3sbitools.utils.config import TrainingConfig

warnings.filterwarnings(
    "ignore",
    message=".*num_workers.*bottleneck.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*LeafSpec.*deprecated.*",
    category=UserWarning,
)


def collate_batch(
    batch: list | tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate a batch, passing through datasets that already stack their own.

    :class:`~mach3sbitools.data_loaders.TrainingDataset` reads a whole batch
    out of its memmaps in one indexing operation, so the default collate's
    unbind-and-restack is wasted work. Datasets without ``__getitems__`` still
    arrive as a list of rows and are stacked normally.

    :param batch: Either a stacked ``(theta, x)`` pair or a list of rows.
    :returns: The stacked ``(theta, x)`` batch.
    """
    if isinstance(batch, list):
        return cast(tuple[torch.Tensor, torch.Tensor], default_collate(batch))
    return batch


class SBIDataModule(L.LightningDataModule):
    """
    Lightning data module over a ``(theta, x)`` map-style dataset.

    Accepts any :class:`~torch.utils.data.Dataset` that returns ``(theta, x)``
    tensor pairs by index — for example a memory-mapped
    :class:`~mach3sbitools.data_loaders.TrainingDataset`, or a pre-loaded
    :class:`~torch.utils.data.TensorDataset` for small datasets.

    Under DDP, Lightning automatically wraps each DataLoader's sampler in a
    ``DistributedSampler``, which partitions the index space across ranks.

    .. note::

        The train/validation split is a random permutation seeded with
        :attr:`split_seed`, so all DDP ranks produce identical index sets.
        Change the seed consistently across ranks or not at all.
    """

    split_seed: int = 42

    def __init__(self, dataset: Dataset, config: TrainingConfig) -> None:
        """
        :param dataset: A map-style ``(theta, x)`` dataset.
        :param config: Training configuration supplying ``validation_fraction``,
            ``batch_size`` and ``num_workers``.
        """
        super().__init__()
        self.dataset = dataset
        self.config = config

        # Specifically still save the batch size
        self.batch_size = config.batch_size

        self.train_dataset: Subset | None = None
        self.val_dataset: Subset | None = None

    def setup(self, stage: str | None = None) -> None:
        """
        Split the dataset into train and validation subsets.

        The split is a seeded random permutation rather than a contiguous
        slice: merged shards arrive grouped by source file, so a contiguous
        validation tail would not be representative of the training set.

        :param stage: Lightning stage hook argument. Unused — the same split
            serves every stage.
        """
        warnings.filterwarnings(
            "ignore",
            message=".*num_workers.*bottleneck.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*LeafSpec.*",
            category=UserWarning,
        )

        n_total = len(cast(Sized, self.dataset))
        n_val = int(n_total * self.config.validation_fraction)
        n_train = n_total - n_val

        generator = torch.Generator().manual_seed(self.split_seed)
        indices = torch.randperm(n_total, generator=generator).tolist()

        self.train_dataset = Subset(self.dataset, indices[:n_train])
        self.val_dataset = Subset(self.dataset, indices[n_train:])

    def _make_dataloader(
        self,
        dataset: Dataset,
        *,
        shuffle: bool,
        drop_last: bool = False,
        batch_multiplier: int = 1,
    ) -> DataLoader:
        """
        Shared factory to avoid duplicating DataLoader kwargs.

        :param dataset: Dataset to wrap.
        :param shuffle: Whether to reshuffle indices every epoch.
        :param drop_last: Whether to drop a trailing partial batch.
        :param batch_multiplier: Scales ``config.batch_size`` for this loader.
        :returns: A configured :class:`~torch.utils.data.DataLoader`.
        """
        use_workers = self.config.num_workers > 0
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size * batch_multiplier,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_batch,
            num_workers=self.config.num_workers,
            # Pinning only buys anything for a host-to-CUDA copy; elsewhere it
            # is a wasted memcpy, and torch warns about it.
            pin_memory=torch.cuda.is_available(),
            persistent_workers=use_workers,
            # The dataset pages rows in from disk, so queue more batches per
            # worker than the default 2 to keep the device fed.
            prefetch_factor=4 if use_workers else None,
        )

    def train_dataloader(self) -> DataLoader:
        """
        Build the shuffled training data loader.

        :returns: Loader over the training split.
        :raises RuntimeError: If :meth:`setup` has not been called.
        """
        if self.train_dataset is None:
            raise RuntimeError("Training set has not been set; call setup() first.")
        return self._make_dataloader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        """
        Build the sequential validation data loader.

        :returns: Loader over the validation split.
        :raises RuntimeError: If :meth:`setup` has not been called.
        """
        if self.val_dataset is None:
            raise RuntimeError("Validation set has not been set; call setup() first.")
        return self._make_dataloader(self.val_dataset, shuffle=False)
