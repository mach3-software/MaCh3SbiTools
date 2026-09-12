"""
Tests for Prior composition and the bounds sanity check.

Device handling, caching and pickling live in test_prior_device.py.
"""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from mach3sbitools.simulator.priors.cyclical_distribution import CyclicalDistribution
from mach3sbitools.simulator.priors.dataclasses import PriorData
from mach3sbitools.simulator.priors.prior import _check_boundary
from mach3sbitools.simulator.priors.truncated_gaussian_distribution import (
    TruncatedGaussianDistribution,
)
from mach3sbitools.utils import get_logger

logger = get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Composition
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestPriorComposition:
    """
    The dummy simulator declares 30 parameters: indices 0-2 flat, index 9
    cyclical (via the `prior` fixture), and the remaining 26 Gaussian.
    """

    def test_one_sub_prior_per_distribution_family(self, prior):
        assert len(prior._priors) == 3

    def test_every_parameter_is_claimed_exactly_once(self, prior, test_consts):
        claimed = sum(mask_map.mask.sum().item() for mask_map in prior._priors)
        assert claimed == test_consts.theta_dim

        overlap = torch.stack([m.mask for m in prior._priors]).sum(dim=0)
        assert torch.all(overlap == 1)

    def test_cyclical_parameter_gets_the_cyclical_distribution(self, prior):
        cyclical = next(
            m for m in prior._priors if isinstance(m.distribution, CyclicalDistribution)
        )
        assert cyclical.mask.sum().item() == 1
        assert cyclical.mask[9].item() is True

    def test_flat_parameters_are_grouped_together(self, prior):
        flat = next(
            m
            for m in prior._priors
            if isinstance(m.distribution, torch.distributions.Uniform)
        )
        assert flat.mask.sum().item() == 3

    def test_remaining_parameters_are_gaussian(self, prior, test_consts):
        gaussian = next(
            m
            for m in prior._priors
            if isinstance(m.distribution, TruncatedGaussianDistribution)
        )
        assert gaussian.mask.sum().item() == test_consts.theta_dim - 4


# ─────────────────────────────────────────────────────────────────────────────
# PriorData slicing
# ─────────────────────────────────────────────────────────────────────────────


def test_prior_data_slicing():
    names = np.array(["a", "b", "c"])
    nominals = torch.tensor([1.0, 2.0, 3.0])
    covariance = torch.tensor([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]])
    lower = torch.tensor([1.0, 2.0, 3.0])
    upper = torch.tensor([4.0, 5.0, 6.0])

    data = PriorData(names, nominals, covariance, lower, upper)
    mask = torch.tensor([True, False, True])
    sliced = data[mask]

    np.testing.assert_array_equal(sliced.parameter_names, ["a", "c"])
    torch.testing.assert_close(sliced.nominals, nominals[mask])
    torch.testing.assert_close(sliced.lower_bounds, lower[mask])
    torch.testing.assert_close(sliced.upper_bounds, upper[mask])
    # The covariance must be sliced on both axes, not just the rows.
    torch.testing.assert_close(
        sliced.covariance_matrix, torch.tensor([[1.0, 3.0], [7.0, 9.0]])
    )


def test_prior_data_slicing_leaves_the_original_alone(test_consts):
    names = np.array(["a", "b"])
    data = PriorData(names, torch.ones(2), torch.eye(2), -torch.ones(2), torch.ones(2))
    data[torch.tensor([True, False])]
    assert len(data.nominals) == 2


# ─────────────────────────────────────────────────────────────────────────────
# _check_boundary
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckBoundary:
    """Warns when a bound sits further than 10σ from the nominal."""

    NOMINAL = torch.tensor([1.0, 2.0])
    ERROR = torch.tensor([0.1, 0.1])
    NAMES = np.array(["param_a", "param_b"])

    def _run(self, lower, upper):
        with patch.object(logger, "warning") as warning:
            _check_boundary(self.NOMINAL, self.ERROR, lower, upper, self.NAMES)
        return warning

    def test_warns_once_per_offending_parameter(self):
        warning = self._run(
            lower=torch.tensor([-100.0, 2.0]), upper=torch.tensor([1.0, 100.0])
        )
        # One header line plus one line per offending parameter.
        assert warning.call_count == 3

    def test_warning_names_the_offending_parameters(self):
        warning = self._run(
            lower=torch.tensor([-100.0, 2.0]), upper=torch.tensor([1.0, 100.0])
        )
        logged = " ".join(str(c) for c in warning.call_args_list)
        assert "param_a" in logged
        assert "param_b" in logged

    def test_silent_when_every_bound_is_close(self):
        warning = self._run(
            lower=torch.tensor([0.5, 1.5]), upper=torch.tensor([1.5, 2.5])
        )
        warning.assert_not_called()

    def test_only_the_offending_parameter_is_reported(self):
        warning = self._run(
            lower=torch.tensor([-100.0, 1.5]), upper=torch.tensor([1.5, 2.5])
        )
        logged = " ".join(str(c) for c in warning.call_args_list)
        assert "param_a" in logged
        assert "param_b" not in logged
