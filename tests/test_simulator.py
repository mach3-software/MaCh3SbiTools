"""
Tests for mach3sbitools.simulator — Prior construction, sampling,
persistence, and the Simulator forward/persistence interface.
"""

import pickle
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from mach3sbitools.simulator.priors.dataclasses import PriorData
from mach3sbitools.simulator.priors.prior import (
    Prior,
    PriorNotFound,
    load_prior,
)
from mach3sbitools.simulator.simulator import Simulator
from mach3sbitools.utils import TorchDeviceHandler, from_feather

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_prior(n: int = 5) -> Prior:
    dh = TorchDeviceHandler()
    data = PriorData(
        parameter_names=np.array([f"p{i}" for i in range(n)]),
        nominals=torch.ones(n),
        covariance_matrix=torch.eye(n),
        lower_bounds=torch.full((n,), -5.0),
        upper_bounds=torch.full((n,), 5.0),
    )
    return Prior(prior_data=data).to(dh.device)


@pytest.fixture
def simulator(dummy_config):
    return Simulator(
        module_name="dummy_simulator",
        class_name="DummySimulator",
        config=dummy_config,
        cyclical_pars=["theta_9"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prior
# ─────────────────────────────────────────────────────────────────────────────


class TestPrior:
    @pytest.mark.slow
    def test_conftest_prior_has_three_distribution_types(self, prior):
        """cyclical + flat + gaussian give 3 sub-distributions."""
        assert len(prior._priors) == 3
        counts = [torch.sum(p.mask).item() for p in prior._priors]
        assert counts == [1, 3, 26]

    def test_samples_are_within_bounds(self):
        prior = _make_prior()
        samples = prior.sample(torch.Size([100]))
        assert torch.all(samples >= prior.prior_data.lower_bounds)
        assert torch.all(samples <= prior.prior_data.upper_bounds)

    def test_prior_data_slicing(self):
        data = PriorData(
            parameter_names=np.array(["a", "b", "c"]),
            nominals=torch.tensor([1.0, 2.0, 3.0]),
            covariance_matrix=torch.eye(3),
            lower_bounds=torch.tensor([0.0, 0.0, 0.0]),
            upper_bounds=torch.tensor([5.0, 5.0, 5.0]),
        )
        mask = torch.tensor([True, False, True])
        sliced = data[mask]
        np.testing.assert_array_equal(sliced.parameter_names, ["a", "c"])
        assert sliced.nominals.shape == (2,)
        assert sliced.covariance_matrix.shape == (2, 2)

    def test_save_and_load_round_trip(self, tmp_path):
        prior = _make_prior()
        out = tmp_path / "prior.pkl"
        prior.save(out)
        loaded = load_prior(out)
        assert isinstance(loaded, Prior)
        assert loaded.n_params == prior.n_params

    def test_load_prior_raises_for_missing_or_wrong_type(self, tmp_path):
        with pytest.raises(PriorNotFound):
            load_prior(tmp_path / "nonexistent.pkl")

        bad = tmp_path / "bad.pkl"
        with bad.open("wb") as f:
            pickle.dump({"not": "a prior"}, f)
        with pytest.raises(PriorNotFound):
            load_prior(bad)


# ─────────────────────────────────────────────────────────────────────────────
# Simulator — forward simulation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestSimulatorForward:
    def test_output_is_non_negative_integers(self, simulator):
        _, x = simulator.simulate(1000)
        assert np.all(x >= 0)
        assert np.all(x == x.astype(int))

    def test_output_mean_near_one(self, simulator):
        """DummySimulator returns ones so Poisson(1) samples should have mean≈1."""
        _, x = simulator.simulate(20_000)
        assert np.all(np.abs(x.mean(axis=0) - 1.0) < 0.05)

    def test_skips_bad_simulations(self, dummy_config):
        sim = Simulator("dummy_simulator", "DummySimulator", config=dummy_config)
        call_count = [0]
        original = sim.simulator_wrapper.simulate

        def flaky(_):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise RuntimeError("bad sim")
            return original(_)

        sim.simulator_wrapper.simulate = flaky
        theta, x = sim.simulate(10)
        assert len(theta) < 10 and len(theta) == len(x)

    def test_returns_empty_arrays_when_all_fail(self, dummy_config):
        sim = Simulator("dummy_simulator", "DummySimulator", config=dummy_config)
        sim.simulator_wrapper.simulate = MagicMock(side_effect=RuntimeError("bad"))
        theta, _ = sim.simulate(5)
        assert len(theta) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Simulator — persistence
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestSimulatorPersistence:
    def test_save_and_reload_feather(self, simulator, tmp_path):
        theta, x = simulator.simulate(10)
        path = tmp_path / "data.feather"
        simulator.save(path, theta, x)
        t, x2 = from_feather(path, simulator.prior.nuisance_filter)
        assert len(t) == len(x2) == 10

    def test_save_data_creates_parquet(self, dummy_config, tmp_path):
        sim = Simulator("dummy_simulator", "DummySimulator", config=dummy_config)
        out = tmp_path / "obs.parquet"
        sim.save_data(out)
        assert out.exists()
