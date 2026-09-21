"""
PyTorch Lightning data module for SBI simulation datasets.

Dataset sharing strategy
------------------------
This module now accepts any map-style ``Dataset`` — including a
memory-mapped, lazily-loading dataset such as
:class:`~mach3sbitools.data_loaders.LazyFeatherDataset` — rather than
requiring a pre-loaded, fully in-RAM ``TensorDataset``.

* Ordering comes from
  :class:`~mach3sbitools.data_loaders.BlockShuffleSampler`, which shuffles
  contiguous blocks rather than rows and partitions blocks across DDP ranks
  itself. The Trainer therefore sets ``use_distributed_sampler=False``; if
  that is ever flipped back on, Lightning replaces this sampler with a
  ``DistributedSampler`` and row-granular access returns.
* When the dataset is backed by memory-mapped, uncompressed feather files,
  ranks on the same node share the OS page cache: identical pages aren't
  duplicated in physical RAM even though each rank/worker opens its own
  ``mmap()``. Across nodes there's no such sharing, but each node still
  only pages in what its own ranks actually touch.
* If a plain in-RAM ``TensorDataset`` is passed instead (e.g. for a small
  dataset that comfortably fits in memory), the same code path works
  unchanged — ``random_split`` and ``DistributedSampler`` only need
  index-level slicing of a map-style ``Dataset``, and don't care whether
  the underlying storage is a tensor or an mmap-backed array.

``num_workers`` should generally be > 0 when the dataset performs lazy
per-row I/O (e.g. :class:`LazyFeatherDataset`), so that disk/page-cache
reads for the next batch overlap with GPU compute on the current one. This
is the opposite of the old advice for a fully RAM-resident
``TensorDataset``, where extra worker processes only added IPC overhead
for no benefit.
"""

from __future__ import annotations

import warnings
from collections.abc import Sized
from typing import cast

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from mach3sbitools.data_loaders.block_sampler import BlockShuffleSampler
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


def _collate_batched(batch):
    """
    Pass through batches that the dataset already collated.

    ``TrainingDataset.__getitems__`` returns ``(theta, x)`` as stacked
    tensors, so there is nothing left to do. Falling back to
    ``default_collate`` keeps this working for any dataset that yields plain
    per-row samples instead (e.g. a ``TensorDataset``).

    Defined at module scope so it survives pickling to spawned workers.
    """
    if isinstance(batch, tuple) and len(batch) == 2 and torch.is_tensor(batch[0]):
        return batch
    return default_collate(batch)


class SBIDataModule(L.LightningDataModule):
    """
    Lightning data module over a ``(theta, x)`` map-style dataset.

    Accepts any :class:`~torch.utils.data.Dataset` that returns
    ``(theta, x)`` tensor pairs by index — for example a lazily-loading,
    memory-mapped :class:`~mach3sbitools.data_loaders.LazyFeatherDataset`,
    or a pre-loaded :class:`~torch.utils.data.TensorDataset` for small
    datasets.

    Train/validation is a deterministic contiguous split -- the first
    ``1 - validation_fraction`` of rows train, the tail validates -- so every
    rank derives identical index sets without needing to agree on a seed.
    Decorrelation is the sampler's job, not the split's.
    """

    def __init__(self, dataset: Dataset, config: TrainingConfig) -> None:
        """
        :param dataset: A map-style ``(theta, x)`` :class:`~torch.utils.data.Dataset`,
            e.g. a :class:`~mach3sbitools.data_loaders.LazyFeatherDataset`
            or a pre-loaded :class:`~torch.utils.data.TensorDataset`.
        :param config: Training configuration supplying ``validation_fraction``
            and ``batch_size``.
        """
        super().__init__()
        self.dataset = dataset
        self.config = config

        # Specifically still save the batch size
        self.batch_size = config.batch_size

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
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

        # `range`, not `torch.arange`: Subset.__getitems__ evaluates
        # `self.indices[i]` once per row per batch, and a tensor returns a
        # 0-d tensor there rather than an int -- thousands of tiny tensor
        # allocations per batch before a single byte is read.
        self.train_dataset = torch.utils.data.Subset(self.dataset, range(n_train))
        self.val_dataset = torch.utils.data.Subset(
            self.dataset, range(n_train, n_total)
        )

    def _make_dataloader(
        self,
        dataset: Dataset,
        *,
        shuffle: bool,
        drop_last: bool = False,
        batch_multiplier: int = 1,
    ) -> DataLoader:
        """Shared factory to avoid duplicating DataLoader kwargs."""
        use_workers = self.config.num_workers > 0
        sampler = BlockShuffleSampler(
            len(cast(Sized, dataset)),
            block_size=self.config.shuffle_block_size,
            shuffle=shuffle,
            seed=self.config.shuffle_seed,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size * batch_multiplier,
            # `sampler` and `shuffle` are mutually exclusive in DataLoader;
            # the sampler owns ordering now.
            sampler=sampler,
            drop_last=drop_last,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=use_workers,
            # Staged bytes are num_workers x prefetch_factor x batch_size x
            # row_bytes, all of it pinned. Large values here are a common
            # cause of host OOM at big batch sizes.
            prefetch_factor=self.config.prefetch_factor if use_workers else None,
            collate_fn=_collate_batched,
        )

    def train_dataloader(self) -> DataLoader:
        """
        Training data loader
        """
        if self.train_dataset is None:
            raise RuntimeError("Training set has not been set; call setup() first.")
        return self._make_dataloader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        """
        Validation data loader
        """
        if self.val_dataset is None:
            raise RuntimeError("Validation set has not been set; call setup() first.")
        # Validation runs under no_grad, so activations are not kept and a
        # much larger batch costs nothing in memory while cutting the number
        # of round trips through the loader.
        return self._make_dataloader(
            self.val_dataset,
            shuffle=False,
            batch_multiplier=self.config.val_batch_multiplier,
        )
