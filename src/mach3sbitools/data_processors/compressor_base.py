from abc import ABC, abstractmethod
from typing import Self

import torch


class CompressorBase(ABC):
    @abstractmethod
    def fit(self, data: torch.Tensor) -> Self:
        """Fit the compressor and return instance"""

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Am I fitted"""

    @abstractmethod
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compress *data*.

        :param data: Shape ``(n, n_features)`` or ``(n_features,)``.
        :returns: Shape ``(n, n_compressed)`` or ``(n_compressed,)``.
        """

    @abstractmethod
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct from compressed representation.

        :param data: Shape ``(n, n_compressed)`` or ``(n_compressed,)``.
        :returns: Shape ``(n, n_features)`` or ``(n_features,)``.
        """

    @property
    @abstractmethod
    def n_components(self) -> int:
        """Dimensionality of the compressed representation."""

    @abstractmethod
    def state_dict(self) -> dict:
        """Serialise to a plain dict of tensors and primitives."""

    @classmethod
    @abstractmethod
    def from_state_dict(cls, state: dict) -> Self:
        """Restore from a state dict."""

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _unsqueeze_if_1d(data: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Add a batch dimension if input is 1D. Returns (data, was_squeezed)."""
        if data.ndim == 1:
            return data.unsqueeze(0), True
        return data, False

    @staticmethod
    def _squeeze_if_needed(data: torch.Tensor, squeeze: bool) -> torch.Tensor:
        return data.squeeze(0) if squeeze else data
