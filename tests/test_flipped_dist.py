"""
Analytical properties of the flipped-uniform prior.

The distribution is uniform on ``[lower, upper]`` and its mirror image
``[-upper, -lower]``, with a forbidden gap around zero. Shape, support and
device behaviour shared with the other priors is in
test_prior_distributions.py.
"""

import numpy as np
import pytest
import torch

from mach3sbitools.simulator.priors.flipped_uniform_distribution import (
    FlippedUniformDistribution,
)

LOWER, UPPER = 1.0, 3.0


@pytest.fixture(scope="session")
def flipped_distribution() -> FlippedUniformDistribution:
    return FlippedUniformDistribution(torch.ones(1), lower=LOWER, upper=UPPER)


@pytest.fixture(scope="session")
def large_samples(flipped_distribution) -> torch.Tensor:
    torch.manual_seed(42)
    return flipped_distribution.sample(torch.Size([50_000]))


# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "lower,upper",
    [(0.0, 3.0), (-1.0, 3.0), (3.0, 3.0), (4.0, 3.0)],
)
def test_rejects_invalid_bounds(lower, upper):
    with pytest.raises(ValueError, match="0 < lower < upper"):
        FlippedUniformDistribution(torch.ones(1), lower=lower, upper=upper)


# ─────────────────────────────────────────────────────────────────────────────
# Analytical properties
# ─────────────────────────────────────────────────────────────────────────────


def test_mean_is_zero_by_symmetry(flipped_distribution):
    torch.testing.assert_close(
        flipped_distribution.mean, torch.zeros(1, dtype=torch.double)
    )


def test_variance_matches_closed_form(flipped_distribution):
    expected = (UPPER**2 + LOWER * UPPER + LOWER**2) / 3.0
    torch.testing.assert_close(
        flipped_distribution.variance,
        torch.full((1,), expected, dtype=torch.double),
    )


@pytest.mark.parametrize("value", [1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
def test_log_prob_is_constant_inside_support(flipped_distribution, value):
    expected = np.log(0.5 / (UPPER - LOWER))
    got = flipped_distribution.log_prob(torch.tensor([value], dtype=torch.double))
    assert got.item() == pytest.approx(expected)


@pytest.mark.parametrize("value", [0.0, 0.5, -0.5, 3.5, -3.5, 100.0])
def test_log_prob_is_negative_infinity_outside_support(flipped_distribution, value):
    got = flipped_distribution.log_prob(torch.tensor([value], dtype=torch.double))
    assert got.item() == -np.inf


def test_density_integrates_to_one(flipped_distribution):
    """Both regions together must carry unit mass."""
    density = np.exp(np.log(0.5 / (UPPER - LOWER)))
    assert 2 * density * (UPPER - LOWER) == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────


def test_samples_lie_in_support(large_samples):
    magnitudes = large_samples.abs()
    assert torch.all(magnitudes >= LOWER)
    assert torch.all(magnitudes <= UPPER)


def test_samples_avoid_the_gap(large_samples):
    assert not torch.any(large_samples.abs() < LOWER)


def test_signs_are_balanced(large_samples):
    positive_fraction = (large_samples > 0).double().mean().item()
    assert positive_fraction == pytest.approx(0.5, abs=0.02)


def test_sample_mean_near_zero(large_samples):
    assert abs(large_samples.mean().item()) < 0.05


def test_sample_variance_matches_analytic(flipped_distribution, large_samples):
    expected = flipped_distribution.variance[0].item()
    assert large_samples.var().item() == pytest.approx(expected, rel=0.05)
