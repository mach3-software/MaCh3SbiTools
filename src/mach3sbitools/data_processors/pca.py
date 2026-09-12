import torch

from mach3sbitools.utils import get_logger

from .compressor_base import CompressorBase

logger = get_logger()


class PCACompressor(CompressorBase):
    """
    Linear dimensionality reduction via randomised PCA.

    Fitting uses :func:`torch.pca_lowrank`, which is approximate but scales
    to the row counts typical of merged simulation sets.
    """

    def __init__(
        self,
        n_components: int,
        subsample: int = 2_000_000,
        niter: int = 4,
    ) -> None:
        """
        :param n_components: Target dimensionality after compression.
        :param subsample: Cap on rows used for fitting; larger inputs are
            randomly subsampled down to this many rows.
        :param niter: Power iterations for the randomised SVD. More is more
            accurate and slower.
        """
        self._n_components = n_components
        self.subsample = subsample
        self.niter = niter

        self.mean: torch.Tensor | None = None
        self.components: torch.Tensor | None = None
        self.explained_variance: torch.Tensor | None = None
        self._n_samples_fit: int = 0
        self._n_features: int = 0

    @property
    def is_fitted(self) -> bool:
        """
        :returns: ``True`` once the principal components have been computed.
        """
        return self.mean is not None

    @property
    def n_components(self) -> int:
        """
        :returns: Dimensionality of the compressed representation.
        """
        return self._n_components

    def fit(self, data: torch.Tensor) -> "PCACompressor":
        """
        Compute the principal components of *data*.

        :param data: Sample of shape ``(n_samples, n_features)``.
        :returns: This instance, fitted.
        :raises ValueError: If ``n_components`` exceeds ``n_features``.
        """
        data = data.float()
        n_samples, n_features = data.shape

        if n_features < self._n_components:
            raise ValueError(
                f"n_components={self._n_components} exceeds n_features={n_features}."
            )

        if n_samples > self.subsample:
            idx = torch.randperm(n_samples)[: self.subsample]
            data = data[idx]
            logger.info(
                f"PCA fitting on {self.subsample:,} subsampled rows "
                f"(total={n_samples:,})"
            )
        else:
            logger.info(f"PCA fitting on full dataset ({n_samples:,} rows)")

        self._n_samples_fit = len(data)
        self._n_features = n_features
        self.mean = data.mean(dim=0)
        centred = data - self.mean

        _, S, V = torch.pca_lowrank(centred, q=self._n_components, niter=self.niter)

        self.components = V.T
        self.explained_variance = (S**2) / (self._n_samples_fit - 1)

        ev_ratio = self.explained_variance_ratio()
        logger.info(
            f"PCA fitted | {n_features} → {self._n_components} components | "
            f"cumulative variance explained: {ev_ratio.sum():.4f} | "
            f"per-component range: [{ev_ratio.min():.4f}, {ev_ratio.max():.4f}]"
        )
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Project *data* onto the fitted principal components.

        The stored components follow *data* onto its device, so a compressor
        fitted on CPU can transform a CUDA batch without the caller moving
        anything by hand.

        :param data: Shape ``(n, n_features)`` or ``(n_features,)``.
        :returns: Shape ``(n, n_components)`` or ``(n_components,)``.
        :raises RuntimeError: If the compressor has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("PCACompressor must be fitted before transform.")

        assert self.components is not None
        assert self.mean is not None

        # 1. Identify where the incoming data lives (CPU or CUDA)
        device = data.device
        dtype = torch.float32

        # 2. Dynamically shift the compressor parameters to match it
        self.mean = self.mean.to(device=device, dtype=dtype)
        self.components = self.components.to(device=device, dtype=dtype)

        # 3. Perform the math safely on the same device
        data, squeezed = self._unsqueeze_if_1d(data.to(dtype=dtype))
        out = (data - self.mean) @ self.components.T
        return self._squeeze_if_needed(out, squeezed)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct an approximation of the original features.

        The reconstruction is lossy: variance dropped at fit time cannot be
        recovered.

        :param data: Shape ``(n, n_components)`` or ``(n_components,)``.
        :returns: Shape ``(n, n_features)`` or ``(n_features,)``.
        :raises RuntimeError: If the compressor has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("PCACompressor must be fitted before inverse_transform.")

        assert self.components is not None
        assert self.mean is not None

        # 1. Identify where the incoming data lives (CPU or CUDA)
        device = data.device
        dtype = torch.float32

        # 2. Dynamically shift the compressor parameters to match it
        self.mean = self.mean.to(device=device, dtype=dtype)
        self.components = self.components.to(device=device, dtype=dtype)

        # 3. Perform the math safely on the same device
        data, squeezed = self._unsqueeze_if_1d(data.to(dtype=dtype))
        out = data @ self.components + self.mean
        return self._squeeze_if_needed(out, squeezed)

    def explained_variance_ratio(self) -> torch.Tensor:
        """
        Fraction of retained variance carried by each component.

        Note this is normalised over the retained components only, so it
        always sums to one regardless of how much variance was discarded.

        :returns: Tensor of shape ``(n_components,)``.
        :raises RuntimeError: If the compressor has not been fitted.
        """
        if self.explained_variance is None:
            raise RuntimeError("PCACompressor is not fitted.")
        return self.explained_variance / self.explained_variance.sum()

    def state_dict(self) -> dict:
        """
        Serialise the fitted components for checkpointing.

        :returns: A dict of tensors and primitives.
        """
        return {
            "type": "pca",
            "n_components": self._n_components,
            "subsample": self.subsample,
            "niter": self.niter,
            "mean": self.mean,
            "components": self.components,
            "explained_variance": self.explained_variance,
            "n_samples_fit": self._n_samples_fit,
            "n_features": self._n_features,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "PCACompressor":
        """
        Restore a fitted compressor from :meth:`state_dict` output.

        :param state: Previously serialised compressor state.
        :returns: A fitted :class:`PCACompressor`.
        """
        obj = cls(
            n_components=state["n_components"],
            subsample=state["subsample"],
            niter=state["niter"],
        )
        obj.mean = state["mean"]
        obj.components = state["components"]
        obj.explained_variance = state["explained_variance"]
        obj._n_samples_fit = state["n_samples_fit"]
        obj._n_features = state["n_features"]
        return obj
