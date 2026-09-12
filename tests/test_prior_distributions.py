"""
The contract every prior sub-distribution must satisfy.

``Prior`` composes these behind one interface and ``sbi`` consumes that
interface, so they have to agree on sample shapes, support handling and
device moves regardless of the maths inside. Distribution-specific analytics
live in the per-distribution test modules.
"""

import numpy as np
import pytest
import torch

from mach3sbitools.simulator.priors.cyclical_distribution import CyclicalDistribution
from mach3sbitools.simulator.priors.flipped_uniform_distribution import (
    FlippedUniformDistribution,
)
from mach3sbitools.simulator.priors.truncated_gaussian_distribution import (
    TruncatedGaussianDistribution,
)

N_PARAMS = 3

# (name, factory, a point far outside the support)
DISTRIBUTIONS = [
    pytest.param(
        lambda: CyclicalDistribution(torch.ones(N_PARAMS)),
        100.0,
        id="cyclical",
    ),
    pytest.param(
        lambda: FlippedUniformDistribution(torch.ones(N_PARAMS), lower=1.0, upper=3.0),
        100.0,
        id="flipped-uniform",
    ),
    pytest.param(
        lambda: TruncatedGaussianDistribution(
            mean=torch.zeros(N_PARAMS),
            covariance=torch.eye(N_PARAMS),
            lower_bounds=-torch.ones(N_PARAMS),
            upper_bounds=torch.ones(N_PARAMS),
        ),
        100.0,
        id="truncated-gaussian",
    ),
]


@pytest.mark.parametrize("factory,far_outside", DISTRIBUTIONS)
class TestDistributionContract:
    @pytest.mark.parametrize(
        "shape,expected",
        [
            (torch.Size([8]), (8, N_PARAMS)),
            (torch.Size([]), (N_PARAMS,)),
            (torch.Size([2, 3]), (2, 3, N_PARAMS)),
        ],
    )
    def test_sample_shape(self, factory, far_outside, shape, expected):
        assert factory().sample(shape).shape == expected

    def test_rsample_matches_sample_shape(self, factory, far_outside):
        distribution = factory()
        assert distribution.rsample(torch.Size([8])).shape == (8, N_PARAMS)

    def test_own_samples_are_in_support(self, factory, far_outside):
        """Every drawn sample must have finite density under the same law."""
        distribution = factory()
        samples = distribution.sample(torch.Size([256]))
        assert torch.all(torch.isfinite(distribution.log_prob(samples)))

    def test_far_outside_the_support_is_negative_infinity(self, factory, far_outside):
        distribution = factory()
        value = torch.full((1, N_PARAMS), far_outside, dtype=torch.double)
        log_prob = distribution.log_prob(value)
        assert torch.all(log_prob == -np.inf)

    def test_variance_is_positive_and_per_parameter(self, factory, far_outside):
        variance = factory().variance
        assert torch.all(variance > 0)

    def test_to_returns_self_and_keeps_device(self, factory, far_outside):
        distribution = factory()
        assert distribution.to("cpu") is distribution
        assert distribution.sample(torch.Size([2])).device.type == "cpu"

    def test_sampling_is_reproducible_under_a_seed(self, factory, far_outside, request):
        """
        Seeding torch must fix the draw.

        TruncatedGaussian samples through SciPy's inverse-CDF, which uses
        numpy's global RNG, so torch.manual_seed does not reach it. Marked
        xfail rather than skipped so it flips the moment that is fixed.
        """
        if "truncated-gaussian" in request.node.callspec.id:
            request.node.add_marker(
                pytest.mark.xfail(
                    strict=True,
                    reason="samples via SciPy, so torch.manual_seed does not apply",
                )
            )

        distribution = factory()
        torch.manual_seed(0)
        first = distribution.sample(torch.Size([16]))
        torch.manual_seed(0)
        second = distribution.sample(torch.Size([16]))
        torch.testing.assert_close(first, second)
