import numpy as np
import torch
from scipy.stats import truncnorm
from torch.distributions import MultivariateNormal

from mach3sbitools.utils import get_logger

logger = get_logger()


class TruncatedGaussianDistribution(MultivariateNormal):
    """
    Bounded multivariate Gaussian with exact sequential-conditional sampling.

    This replaces the TransformedDistribution approach, avoiding exploding
    tails (inverse-Jacobian issues) and preserving the true covariance matrix.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        covariance: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
    ) -> None:
        # ── Symmetrise + Cholesky with jitter fallback ─────────────────────
        cov = (covariance + covariance.T) / 2.0
        jitter_amount = 1e-9
        chol: torch.Tensor | None = None

        for attempt in range(6):
            try:
                chol = torch.linalg.cholesky(cov)
                break
            except Exception:
                logger.warning(
                    "Covariance not positive definite (attempt %d); adding jitter",
                    attempt + 1,
                )
                jitter = jitter_amount * torch.eye(
                    len(mean), dtype=cov.dtype, device=cov.device
                )
                jitter[torch.diag(cov) * 10 < jitter] = 0
                cov += jitter

                jitter *= 10

        if chol is None:
            raise ValueError(
                "Covariance could not be made positive definite after 6 attempts."
            )

        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

        # Cache numpy copies because scipy.stats requires them
        self._L_np: np.ndarray = chol.detach().cpu().numpy()
        self._mean_np: np.ndarray = mean.detach().cpu().numpy()
        self._lower_np: np.ndarray = lower_bounds.detach().cpu().numpy()
        self._upper_np: np.ndarray = upper_bounds.detach().cpu().numpy()

        super().__init__(loc=mean, scale_tril=chol)

    # ── Bounds helper ───────────────────────────────────────────────────────

    def in_bounds(self, value: torch.Tensor) -> torch.Tensor:
        """Check if values are strictly within the bounding box."""
        return torch.all(
            (value >= self._lower_bounds) & (value <= self._upper_bounds), dim=-1
        )

    # ── Probability interface ───────────────────────────────────────────────

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Log probability density.
        Returns the true Gaussian log-density inside bounds, and -inf outside.
        """
        in_b = self.in_bounds(value)
        lp = torch.full(
            value.shape[:-1], float("-inf"), dtype=value.dtype, device=value.device
        )
        if in_b.any():
            lp[in_b] = super().log_prob(value[in_b]).to(dtype=lp.dtype)
        return lp

    # ── Exact sampler ───────────────────────────────────────────────────────

    def sample(
        self,
        sample_shape: torch.Size | list[int] | tuple[int, ...] = torch.Size(),
    ) -> torch.Tensor:
        """
        Draw exact samples via sequential conditional sampling.

        Iterates over dimensions. At each step, the conditional bounds are
        computed from already-sampled variables. A batch of 1-D truncated
        normals is then sampled simultaneously via SciPy's inverse-CDF.
        """
        n_samples = int(np.prod(sample_shape)) if sample_shape else 1
        d = self._mean_np.shape[0]

        L = self._L_np  # (d, d) lower-triangular
        mu = self._mean_np  # (d,)
        lower = self._lower_np  # (d,)
        upper = self._upper_np  # (d,)

        # z holds the *whitened* coordinates: x = L z + mu
        z = np.empty((n_samples, d), dtype=np.float64)

        for i in range(d):
            # ── Conditional mean shift ────────────────────────────────────
            if i == 0:
                shift = 0.0
            else:
                shift = z[:, :i] @ L[i, :i]  # (n,) — BLAS-2 gemv

            sigma_i = L[i, i]  # conditional std (constant)
            mu_i = mu[i] + shift  # (n,) conditional mean

            # ── Truncation bounds for x_i ─────────────────────────────────
            a = (lower[i] - mu_i) / sigma_i
            b = (upper[i] - mu_i) / sigma_i

            # ── Sample x_i from 1-D truncated normal ─────────────────────
            x_i = truncnorm.rvs(a, b, loc=mu_i, scale=sigma_i, size=n_samples)

            # ── Store whitened value for next iterations ──────────────────
            z[:, i] = (x_i - mu[i] - (shift if i > 0 else 0.0)) / sigma_i

        # Recover x in original space: x = L z + mu
        x_np = (L @ z.T).T + mu

        result = torch.from_numpy(x_np).to(dtype=self.loc.dtype, device=self.loc.device)

        # Match torch.distributions batch shape convention
        if not sample_shape:
            return result.squeeze(0)
        return result.reshape(*sample_shape, d)

    def rsample(
        self,
        sample_shape: torch.Size | list[int] | tuple[int, ...] = torch.Size(),
    ) -> torch.Tensor:
        """
        Reparameterised sample (delegates to sample).

        Provided to satisfy the PyTorch Distribution interface expected by sbi.
        Note: The Scipy sampling path is not strictly differentiable.
        """
        return self.sample(sample_shape)
