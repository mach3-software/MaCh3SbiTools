"""
Compression interfaces and implementations for SBI dimensionality reduction.

Both the parameter space (theta) and observable space (x) can be compressed
independently before being fed to the density estimator, and decompressed
transparently on the way out.

Usage
-----
Fit a PCA compressor on training data and attach it to the handler::

    theta_compressor = PCACompressor(n_components=20)
    x_compressor     = PCACompressor(n_components=10)

    handler.fit_compressors(
        theta_compressor=theta_compressor,
        x_compressor=x_compressor,
    )

    # Then train as normal — compression is applied automatically.
    handler.train_posterior(config, model_config=config)

After loading a checkpoint, compressors are restored automatically::

    handler.load_posterior(checkpoint_path)
    samples = handler.sample_posterior(10_000, x_observed)
    # samples are in the *original* theta space — decompression is transparent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Compressor(ABC):
    """
    Interface for a paired compress / decompress transform.

    Both methods accept and return :class:`torch.Tensor` objects so they can
    sit directly in the data pipeline without dtype juggling.  Implementations
    are responsible for storing all state needed to serialise themselves via
    :meth:`state_dict` / :meth:`load_state_dict`.
    """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Return ``True`` once :meth:`fit` has been called successfully."""

    @abstractmethod
    def fit(self, data: torch.Tensor) -> "Compressor":
        """
        Fit the compressor to *data* and return ``self`` for chaining.

        :param data: 2-D tensor of shape ``(n_samples, n_features)``.
        :returns: ``self``
        """

    @abstractmethod
    def compress(self, data: torch.Tensor) -> torch.Tensor:
        """
        Map *data* from the original space to the compressed space.

        :param data: Tensor of shape ``(n_samples, n_features)``.
        :returns: Tensor of shape ``(n_samples, n_components)``.
        """

    @abstractmethod
    def decompress(self, data: torch.Tensor) -> torch.Tensor:
        """
        Map *data* from the compressed space back to the original space.

        For lossy methods (e.g. PCA) this is the best linear reconstruction,
        not an exact inverse.

        :param data: Tensor of shape ``(n_samples, n_components)``.
        :returns: Tensor of shape ``(n_samples, n_features)``.
        """

    @abstractmethod
    def state_dict(self) -> dict:
        """
        Return all fitted state as a plain dict of tensors / scalars.

        The dict must be sufficient to fully restore the compressor via
        :meth:`load_state_dict` without re-fitting.
        """

    @abstractmethod
    def load_state_dict(self, state: dict) -> "Compressor":
        """
        Restore fitted state from *state* and return ``self``.

        :param state: Dict previously returned by :meth:`state_dict`.
        """

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def input_dim(self) -> Optional[int]:
        """Original feature dimension, or ``None`` if not yet fitted."""
        return None

    @property
    def output_dim(self) -> Optional[int]:
        """Compressed dimension, or ``None`` if not yet fitted."""
        return None

    def __repr__(self) -> str:  # pragma: no cover
        fitted = "fitted" if self.is_fitted else "unfitted"
        dims = (
            f"{self.input_dim}→{self.output_dim}" if self.is_fitted else "?"
        )
        return f"{self.__class__.__name__}({dims}, {fitted})"


# ---------------------------------------------------------------------------
# Identity (no-op) compressor
# ---------------------------------------------------------------------------


class IdentityCompressor(Compressor):
    """
    Pass-through compressor — compress and decompress are both identity maps.

    Useful as a default so the rest of the codebase never has to branch on
    ``compressor is None``.
    """

    @property
    def is_fitted(self) -> bool:
        return True

    def fit(self, data: torch.Tensor) -> "IdentityCompressor":
        return self

    def compress(self, data: torch.Tensor) -> torch.Tensor:
        return data

    def decompress(self, data: torch.Tensor) -> torch.Tensor:
        return data

    def state_dict(self) -> dict:
        return {"type": "identity"}

    def load_state_dict(self, state: dict) -> "IdentityCompressor":
        return self


# ---------------------------------------------------------------------------
# PCA compressor
# ---------------------------------------------------------------------------


class PCACompressor(Compressor):
    """
    Linear PCA compressor using a truncated SVD of the (centred) training data.


    :param n_components: Number of principal components to retain.  If
        ``None``, all components are kept (i.e. a full, lossless PCA).
    :param whiten: If ``True``, divide the projected coordinates by the square
        root of their variance (i.e. standardise in PCA space).  This can
        help density estimators that are sensitive to input scale differences
        across components.
    """

    def __init__(
        self,
        n_components: int | None = None,
        whiten: bool = False,
    ) -> None:
        self.n_components = n_components
        self.whiten = whiten

        self._mean: torch.Tensor | None = None
        self._components: torch.Tensor | None = None   # (n_components, n_features)
        self._explained_variance: torch.Tensor | None = None  # (n_components,)
        self._input_dim: int | None = None

    # ── Compressor interface ──────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._components is not None

    @property
    def input_dim(self) -> int | None:
        return self._input_dim

    @property
    def output_dim(self) -> int | None:
        if self._components is None:
            return None
        return self._components.shape[0]

    def fit(self, data: torch.Tensor) -> "PCACompressor":
        """
        Compute the PCA basis from *data*.

        :param data: Float tensor of shape ``(n_samples, n_features)``.
        :returns: ``self``
        :raises ValueError: If ``n_components`` exceeds ``min(n_samples, n_features)``.
        """
        if data.ndim != 2:
            raise ValueError(f"Expected 2-D data, got shape {tuple(data.shape)}")

        n_samples, n_features = data.shape
        k = self.n_components or min(n_samples, n_features)

        if k > min(n_samples, n_features):
            raise ValueError(
                f"n_components={k} exceeds min(n_samples={n_samples}, "
                f"n_features={n_features})."
            )

        # Centre
        mean = data.mean(dim=0)
        centred = data - mean

        # Economy SVD:  centred = U @ diag(S) @ Vh
        # We keep only the top-k right singular vectors (rows of Vh).
        # torch.linalg.svd with full_matrices=False gives shapes:
        #   U  : (n_samples, min(n,p))
        #   S  : (min(n,p),)
        #   Vh : (min(n,p), n_features)
        _, S, Vh = torch.linalg.svd(centred, full_matrices=False)

        components = Vh[:k]                    # (k, n_features)
        explained_var = (S[:k] ** 2) / (n_samples - 1)

        # Store on CPU for device-agnostic serialisation
        self._mean = mean.cpu()
        self._components = components.cpu()
        self._explained_variance = explained_var.cpu()
        self._input_dim = n_features

        return self

    def compress(self, data: torch.Tensor) -> torch.Tensor:
        """
        Project *data* onto the principal components.

        :param data: Tensor of shape ``(n_samples, n_features)``.
        :returns: Tensor of shape ``(n_samples, n_components)``.
        """
        self._check_fitted()
        mean = self._mean.to(data.device)           # type: ignore[union-attr]
        components = self._components.to(data.device)   # type: ignore[union-attr]

        z = (data - mean) @ components.T            # (n_samples, n_components)

        if self.whiten:
            std = self._explained_variance.to(data.device).sqrt().clamp(min=1e-8)  # type: ignore[union-attr]
            z = z / std

        return z

    def decompress(self, data: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the original-space representation from PCA coordinates.

        :param data: Tensor of shape ``(n_samples, n_components)``.
        :returns: Tensor of shape ``(n_samples, n_features)``.
        """
        self._check_fitted()
        mean = self._mean.to(data.device)               # type: ignore[union-attr]
        components = self._components.to(data.device)   # type: ignore[union-attr]

        z = data
        if self.whiten:
            std = self._explained_variance.to(data.device).sqrt().clamp(min=1e-8)  # type: ignore[union-attr]
            z = z * std

        return z @ components + mean

    # ── Serialisation ─────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        self._check_fitted()
        return {
            "type": "pca",
            "n_components": self.n_components,
            "whiten": self.whiten,
            "mean": self._mean,
            "components": self._components,
            "explained_variance": self._explained_variance,
            "input_dim": self._input_dim,
        }

    def load_state_dict(self, state: dict) -> "PCACompressor":
        self.n_components = state["n_components"]
        self.whiten = state["whiten"]
        self._mean = state["mean"]
        self._components = state["components"]
        self._explained_variance = state["explained_variance"]
        self._input_dim = state["input_dim"]
        return self

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def explained_variance_ratio(self) -> torch.Tensor | None:
        """
        Fraction of total variance captured by each component.

        Returns ``None`` if not yet fitted.  Useful for choosing
        ``n_components`` before training::

            compressor.fit(theta_tensor)
            ratios = compressor.explained_variance_ratio
            cumulative = ratios.cumsum(0)
            # pick n_components where cumulative > 0.99
        """
        if self._explained_variance is None:
            return None
        total = self._explained_variance.sum()
        return self._explained_variance / total.clamp(min=1e-12)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted yet. "
                "Call fit() on training data before compress() / decompress()."
            )


# ---------------------------------------------------------------------------
# Registry helpers — used by checkpoint loading
# ---------------------------------------------------------------------------

_COMPRESSOR_REGISTRY: dict[str, type[Compressor]] = {
    "identity": IdentityCompressor,
    "pca": PCACompressor,
}


def compressor_from_state_dict(state: dict) -> Compressor:
    """
    Reconstruct a :class:`Compressor` from a plain state dict.

    The ``"type"`` key (written by every :meth:`~Compressor.state_dict`
    implementation) is used to look up the concrete class.

    :param state: Dict previously returned by :meth:`Compressor.state_dict`.
    :raises KeyError: If the ``"type"`` value is not in the registry.
    """
    compressor_type = state.get("type", "identity")
    cls = _COMPRESSOR_REGISTRY.get(compressor_type)
    if cls is None:
        raise KeyError(
            f"Unknown compressor type '{compressor_type}'. "
            f"Known types: {sorted(_COMPRESSOR_REGISTRY)}"
        )
    return cls().load_state_dict(state)