"""
PyTorch Lightning data module for SBI simulation datasets.

Dataset sharing strategy
------------------------
This module now accepts any map-style ``Dataset`` — including a
memory-mapped, lazily-loading dataset such as
:class:`~mach3sbitools.data_loaders.LazyFeatherDataset` — rather than
requiring a pre-loaded, fully in-RAM ``TensorDataset``.

* Under DDP, Lightning's built-in ``DistributedSampler`` (activated
  automatically when ``strategy="ddp"``) still gives each rank a disjoint
  slice of indices, so every GPU only touches its own share of rows.
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

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split

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


class SBIDataModule(L.LightningDataModule):
    """
    Lightning data module over a ``(theta, x)`` map-style dataset.

    Accepts any :class:`~torch.utils.data.Dataset` that returns
    ``(theta, x)`` tensor pairs by index — for example a lazily-loading,
    memory-mapped :class:`~mach3sbitools.data_loaders.LazyFeatherDataset`,
    or a pre-loaded :class:`~torch.utils.data.TensorDataset` for small
    datasets.

    Under DDP, Lightning automatically wraps each DataLoader's sampler in a
    ``DistributedSampler``, which partitions the index space across ranks.

    .. note::

        The random split uses a fixed seed of ``42`` so that all DDP ranks
        produce identical train / validation index sets.  If you change this
        seed, change it consistently across all ranks.
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
        n_val = int(len(self.dataset) * self.config.validation_fraction)
        n_train = len(self.dataset) - n_val
        self.train_dataset, self.val_dataset = random_split(
            self.dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
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
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size * batch_multiplier,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=use_workers,
            prefetch_factor=5 if use_workers else None,
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
        return self._make_dataloader(
            self.val_dataset, shuffle=False, batch_multiplier=4
        )