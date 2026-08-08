"""
PyTorch Lightning data module for SBI simulation datasets.

Dataset sharing strategy
------------------------
This module accepts any :class:`~torch.utils.data.Dataset`; the right
choice of ``num_workers`` depends on which kind is passed in.

* **In-memory (** :class:`~torch.utils.data.TensorDataset` **)** — the
  tensors live in **CPU RAM** and are **not copied per DDP rank**.
  Lightning's built-in ``DistributedSampler`` (activated automatically
  when ``strategy="ddp"``) gives each rank a disjoint slice of indices, so
  every GPU reads only its own share from the shared tensor without any
  inter-process data replication. ``num_workers=0`` is usually best here —
  spawning worker processes for an already-RAM-resident tensor only adds
  IPC overhead.
* **Streaming (** :class:`~mach3sbitools.data_loaders.StreamingFeatherDataset`
  **)** — samples are read from disk on demand, so ``num_workers > 0`` is
  what lets shard I/O for the *next* batch overlap with training on the
  current one. ``DistributedSampler`` still works identically either way
  (it only partitions indices, not data), so DDP needs no special handling
  for either dataset kind.
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
    Lightning data module over a ``(theta, x)`` dataset.

    Works with either an in-memory :class:`~torch.utils.data.TensorDataset`
    (produced by
    :meth:`~mach3sbitools.data_loaders.TrainingDataset.to_tensor_dataset`)
    or a disk-backed
    :class:`~mach3sbitools.data_loaders.StreamingFeatherDataset` — see the
    module docstring for how ``num_workers`` should differ between the two.

    Under DDP, Lightning automatically wraps each DataLoader's sampler in a
    ``DistributedSampler``, which partitions the index space across ranks —
    this requires nothing dataset-specific: an in-memory dataset's shared
    CPU tensor means each rank reads its own slice with no data copied
    between processes, while a streaming dataset simply reads its slice's
    shards from disk directly, independently per rank.

    .. note::

        The random split uses a fixed seed of ``42`` so that all DDP ranks
        produce identical train / validation index sets.  If you change this
        seed, change it consistently across all ranks.
    """

    def __init__(self, dataset: Dataset, config: TrainingConfig) -> None:
        """
        :param dataset: A ``(theta, x)`` :class:`~torch.utils.data.Dataset` —
            either a pre-loaded :class:`~torch.utils.data.TensorDataset` or a
            :class:`~mach3sbitools.data_loaders.StreamingFeatherDataset`.
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
