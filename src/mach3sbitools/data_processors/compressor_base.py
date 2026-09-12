from abc import ABC, abstractmethod
from typing import Self

import torch


class CompressorBase(ABC):
    """
    Interface for reversible dimensionality reduction of theta or x.

    A compressor is fitted once on training data, then applied to every
    batch during training and to every observation at inference time, so
    both sides of the pipeline live in the same space.
    """

    @abstractmethod
    def fit(self, data: torch.Tensor) -> Self:
        """
        Fit the compressor on *data*.

        :param data: Training sample of shape ``(n, n_features)``.
        :returns: This instance, fitted.
        """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """
        :returns: ``True`` once :meth:`fit` has completed.
        """

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
        """
        :returns: Dimensionality of the compressed representation.
        """

    @abstractmethod
    def state_dict(self) -> dict:
        """
        Serialise to a plain dict of tensors and primitives.

        :returns: A dict suitable for embedding in a checkpoint.
        """

    @classmethod
    @abstractmethod
    def from_state_dict(cls, state: dict) -> Self:
        """
        Restore a fitted compressor from :meth:`state_dict` output.

        :param state: Previously serialised compressor state.
        :returns: A fitted compressor.
        """

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _unsqueeze_if_1d(data: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """
        Add a batch dimension if *data* is a single unbatched row.

        :param data: Input tensor, 1D or 2D.
        :returns: Tuple of ``(data, was_unsqueezed)``.
        """
        if data.ndim == 1:
            return data.unsqueeze(0), True
        return data, False

    @staticmethod
    def _squeeze_if_needed(data: torch.Tensor, squeeze: bool) -> torch.Tensor:
        """
        Undo :meth:`_unsqueeze_if_1d`.

        :param data: Tensor with a leading batch dimension.
        :param squeeze: Whether that dimension was added artificially.
        :returns: Tensor matching the caller's original rank.
        """
        return data.squeeze(0) if squeeze else data
