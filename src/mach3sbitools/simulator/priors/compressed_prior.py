"""
A thin torch.distributions.Distribution wrapper that presents the *compressed*
(PCA-transformed) parameter space to sbi's DirectPosterior, while proxying
log_prob and support checks back through the original prior (after
decompressing).
"""

from __future__ import annotations

import torch
from torch.distributions import Distribution, constraints

from mach3sbitools.data_processors import CompressorBase
from mach3sbitools.simulator.priors import Prior


class CompressedPriorWrapper(Distribution):
    """
    Wraps a :class:`~mach3sbitools.simulator.priors.Prior` so that sbi sees
    the **compressed** parameter space.

    sbi's ``DirectPosterior`` consults the prior for two things during
    sampling:

    1. ``prior.log_prob(theta)``  — used to compute the posterior log-prob.
    2. ``prior.support.check(theta)`` — used to reject out-of-bounds samples.

    When a theta compressor is active the density estimator produces samples
    in compressed space, so both checks must be done in that space.  This
    wrapper intercepts both and:

    * Decompresses ``theta_compressed`` → ``theta_original`` before
      forwarding to the real prior.
    * Exposes a ``support`` that operates on compressed vectors.

    :param prior: The original :class:`~mach3sbitools.simulator.priors.Prior`.
    :param theta_compressor: Fitted compressor that maps
      ``(n, n_params) → (n, n_components)``.
    """

    arg_constraints: dict = {}
    has_rsample = False

    def __init__(
        self,
        prior: Prior,
        theta_compressor: CompressorBase,
    ) -> None:
        self._prior = prior
        self._compressor = theta_compressor

        n_components = theta_compressor.n_components
        super().__init__(
            batch_shape=torch.Size(),
            event_shape=torch.Size([n_components]),
            validate_args=False,
        )

    # ------------------------------------------------------------------
    # Core interface consumed by sbi
    # ------------------------------------------------------------------
    def log_prob(self, theta_compressed: torch.Tensor) -> torch.Tensor:
        """
        Decompress → evaluate original prior log-prob.
        """
        # 1. Track the incoming device
        device = theta_compressed.device
        
        # 2. Safely ensure the underlying prior matches this device
        if hasattr(self._prior, "to"):
            self._prior = self._prior.to(device)
            
        theta_orig = self._compressor.inverse_transform(
            theta_compressed.to(torch.float32)
        )
        return self._prior.log_prob(theta_orig.to(dtype=torch.double, device=device))


    def sample(
        self,
        sample_shape: torch.Size | tuple[int, ...] = torch.Size(),
    ) -> torch.Tensor:
        """
        Sample from the original prior, then compress.

        Used by sbi when it needs prior proposals (e.g. for leakage
        correction).
        """
        theta_orig = self._prior.sample(torch.Size(sample_shape))
        return self._compressor.transform(theta_orig.to(torch.float32))

    @property
    def mean(self) -> torch.Tensor:
        """Compressed mean (compressed nominal values)."""
        return self._compressor.transform(
            self._prior.mean.unsqueeze(0).to(torch.float32)
        ).squeeze(0)

    # ------------------------------------------------------------------
    # Support constraint — operates in compressed space
    # ------------------------------------------------------------------

    @property
    def support(self) -> constraints.Constraint:
        """
        A constraint that decompresses ``theta`` before delegating to the
        original prior's support check.
        """
        compressor = self._compressor
        original_prior = self._prior

        class _DecompressedConstraint(constraints.Constraint):
            is_discrete = False
            event_dim = 1

            def check(self_, value: torch.Tensor) -> torch.Tensor:  # noqa: N805
                # value : (..., n_components)
                orig = compressor.inverse_transform(value.to(torch.float32))
                # prior.check_bounds expects (n_samples, n_params)
                if orig.dim() == 1:
                    return original_prior.check_bounds(orig.unsqueeze(0)).squeeze(0)
                return original_prior.check_bounds(orig)

        return _DecompressedConstraint()

    # ------------------------------------------------------------------
    # Convenience passthrough
    # ------------------------------------------------------------------

    @property
    def prior_data(self):
        """Expose the underlying prior_data so sbi can read parameter metadata."""
        return self._prior.prior_data