"""
Flipped uniform (bimodal) prior distribution.

Implements a custom :class:`torch.distributions.Distribution` for parameters
whose prior support consists of **two disjoint symmetric regions**:

    [lower, upper]  +  [-upper, -lower]

Each region carries equal probability mass (½ each), so the overall PDF is:

.. math::

    p(\\theta) = \\frac{1}{2(\\text{upper} - \\text{lower})},
    \\quad \\theta \\in [\\text{lower}, \\text{upper}]
               \\cup [-\\text{upper}, -\\text{lower}]

and zero elsewhere.  Sampling is done by first drawing the region
(Bernoulli(0.5)) and then drawing uniformly within it.
"""

from typing import cast

import torch
import torch.distributions


class FlippedUniformDistribution(torch.distributions.Distribution):
    """
    Bimodal uniform prior over two symmetric disjoint intervals.

    The support is ``[lower, upper] + [-upper, -lower]`` with equal mass
    on each half.  All parameters in a single instance share the same
    ``lower`` and ``upper`` scalars; they differ only in the *number* of
    parameters tracked (i.e. the event size driven by ``nominals``).

    :param nominals: Nominal values, one per parameter.  Used only to fix
        the event shape; the values themselves are not used in the PDF.
    :param lower: Positive lower edge of the positive region.  Must satisfy
        ``0 < lower < upper``.
    :param upper: Positive upper edge of the positive region.
    :raises ValueError: If ``lower`` or ``upper`` violate the ordering
        constraint ``0 < lower < upper``.
    """

    def __init__(
        self,
        nominals: torch.Tensor,
        lower: float,
        upper: float,
    ) -> None:
        if lower <= 0 or upper <= lower:
            raise ValueError(
                f"FlippedUniformDistribution requires 0 < lower < upper, "
                f"got lower={lower}, upper={upper}."
            )

        self.nominals = nominals
        self.device = nominals.device
        self._lower = torch.tensor(lower, dtype=torch.double, device=self.device)
        self._upper = torch.tensor(upper, dtype=torch.double, device=self.device)

        print(self._upper, self._lower)

        self._width = self._upper - self._lower  # width of one region
        self._pdf_val = 0.5 / self._width  # constant density on each region

        super().__init__(
            batch_shape=torch.Size(),
            event_shape=torch.Size([len(nominals)]),
            validate_args=False,
        )

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def mean(self) -> torch.Tensor:
        """Mean of the distribution.

        By symmetry (equal mass on positive and negative regions) the mean
        is exactly zero, regardless of ``lower`` and ``upper``.
        """
        return torch.zeros(len(self.nominals), dtype=torch.double, device=self.device)

    @property
    def variance(self) -> torch.Tensor:
        """
        Per-parameter variance of the distribution.

        For a distribution uniform on ``[a, b] + [-b, -a]`` the variance is:

        .. math::

            \\text{Var} = \\frac{(b^2 + ab + a^2)}{3}

        which equals the second moment (the first moment is zero by symmetry).
        """
        a, b = self._lower, self._upper
        var_scalar = (b**2 + a * b + a**2) / 3.0
        return cast(
            torch.Tensor, var_scalar.expand(len(self.nominals)).to(torch.double)
        )

    # ── Probability interface ──────────────────────────────────────────────────

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log probability density at *value*.

        Returns ``log(0.5 / width)`` for values inside either region and
        ``-inf`` elsewhere.

        :param value: Input tensor of arbitrary shape.
        :returns: Log-density tensor, same shape as *value*.
        """
        in_pos = (value >= self._lower) & (value <= self._upper)
        in_neg = (value >= -self._upper) & (value <= -self._lower)
        in_support = in_pos | in_neg

        log_p = torch.full(
            value.shape,
            float("-inf"),
            dtype=torch.double,
            device=self.device,
        )
        log_p[in_support] = torch.log(self._pdf_val).to(torch.double)
        return log_p

    # ── Sampling ───────────────────────────────────────────────────────────────

    def sample(
        self,
        sample_shape: torch.Size | list[int] | tuple[int, ...] = torch.Size(),
    ) -> torch.Tensor:
        """
        Draw samples from the bimodal uniform distribution.

        Each sample independently:

        1. Selects the positive or negative region with probability ½ each.
        2. Draws uniformly from the chosen region.

        :param sample_shape: Desired batch shape.
        :returns: Sampled tensor of shape ``(*sample_shape, event_size)``.
        """
        sample_shape = torch.Size(sample_shape)
        event_size = len(self.nominals)

        # Batch count
        n = int(sample_shape.numel()) if sample_shape else 1

        # Draw uniform samples within [lower, upper]
        u = torch.rand(n, event_size, dtype=torch.double, device=self.device)
        magnitudes = self._lower + u * self._width

        # Assign sign: +1 or -1 with equal probability
        signs = torch.where(
            torch.rand(n, event_size, device=self.device) < 0.5,
            torch.ones(n, event_size, dtype=torch.double, device=self.device),
            -torch.ones(n, event_size, dtype=torch.double, device=self.device),
        )

        samples = signs * magnitudes

        if not sample_shape:
            return samples.squeeze(0)
        return samples.reshape(*sample_shape, event_size)

    def rsample(
        self,
        sample_shape: torch.Size | list[int] | tuple[int, ...] = torch.Size(),
    ) -> torch.Tensor:
        """
        Reparameterised sample (delegates to :meth:`sample`).

        The sign-selection step is not differentiable, so this is not a true
        reparameterisation; it is provided only to satisfy the
        :class:`torch.distributions.Distribution` interface expected by
        ``sbi``.

        :param sample_shape: Desired batch shape.
        :returns: Sampled tensor of shape ``(*sample_shape, event_size)``.
        """
        return self.sample(sample_shape)
