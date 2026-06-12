import torch

from mach3sbitools.utils import get_logger

from .compressor_base import CompressorBase

logger = get_logger()


class PCACompressor(CompressorBase):
    def __init__(
        self,
        n_components: int,
        subsample: int = 2_000_000,
        niter: int = 4,
    ) -> None:
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
        return self.mean is not None

    @property
    def n_components(self) -> int:
        return self._n_components

    def fit(self, data: torch.Tensor) -> "PCACompressor":
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
        if not self.is_fitted:
            raise RuntimeError("PCACompressor must be fitted before transform.")

        assert self.components is not None
        assert self.mean is not None

        data, squeezed = self._unsqueeze_if_1d(data.float())
        out = (data - self.mean) @ self.components.T
        return self._squeeze_if_needed(out, squeezed)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("PCACompressor must be fitted before inverse_transform.")

        assert self.components is not None
        assert self.mean is not None

        data, squeezed = self._unsqueeze_if_1d(data.float())
        out = data @ self.components + self.mean
        return self._squeeze_if_needed(out, squeezed)

    def explained_variance_ratio(self) -> torch.Tensor:
        if self.explained_variance is None:
            raise RuntimeError("PCACompressor is not fitted.")
        return self.explained_variance / self.explained_variance.sum()

    def state_dict(self) -> dict:
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
