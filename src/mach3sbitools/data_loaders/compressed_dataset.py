"""
On-the-fly compression wrapper for map-style ``(theta, x)`` datasets.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from mach3sbitools.data_processors import CompressorBase


class CompressedDataset(Dataset):
    """
    Wraps a base ``(theta, x)`` map-style dataset and applies fitted
    compressors per-item at read time.

    This is the streaming-friendly counterpart to eagerly transforming a
    fully-materialized ``TensorDataset`` in one shot: it works equally well
    whether ``base_dataset`` is a small in-RAM ``TensorDataset`` or a
    memory-mapped, lazily-loading :class:`~mach3sbitools.data_loaders.TrainingDataset`
    covering far more data than fits in RAM.

    :param base_dataset: Any map-style dataset returning ``(theta, x)`` pairs.
    :param theta_compressor: Fitted compressor applied to ``theta``, or
        ``None`` to leave it uncompressed.
    :param x_compressor: Fitted compressor applied to ``x``, or ``None`` to
        leave it uncompressed.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        theta_compressor: CompressorBase | None = None,
        x_compressor: CompressorBase | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.theta_compressor = theta_compressor
        self.x_compressor = x_compressor

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        theta, x = self.base_dataset[idx]

        if self.theta_compressor is not None:
            theta = self.theta_compressor.transform(theta)
        if self.x_compressor is not None:
            x = self.x_compressor.transform(x)

        return theta, x
