"""
Tests for Prior device ownership, caching and pickle round-trips.

The prior holds tensors, sub-distributions and masks that must all agree on
one device, and it is routinely pickled on one machine and loaded on another.
"""

import pickle

import numpy as np
import pytest
import torch

from mach3sbitools.simulator.priors.dataclasses import PriorData
from mach3sbitools.simulator.priors.prior import Prior, PriorNotFound, load_prior
from mach3sbitools.utils import get_device

N_PARAMS = 6


@pytest.fixture
def prior_data() -> PriorData:
    return PriorData(
        parameter_names=np.array([f"p{i}" for i in range(N_PARAMS)]),
        nominals=torch.zeros(N_PARAMS),
        covariance_matrix=torch.eye(N_PARAMS),
        lower_bounds=-torch.ones(N_PARAMS),
        upper_bounds=torch.ones(N_PARAMS),
    )


@pytest.fixture
def simple_prior(prior_data) -> Prior:
    return Prior(prior_data, flat_msk=[True] * N_PARAMS)


# ─────────────────────────────────────────────────────────────────────────────
# Device consistency
# ─────────────────────────────────────────────────────────────────────────────


class TestPriorDeviceConsistency:
    def test_device_matches_the_global_default(self, simple_prior):
        assert simple_prior.device == get_device()

    def test_every_tensor_lives_on_the_reported_device(self, simple_prior):
        device = simple_prior.device
        assert simple_prior.prior_data.nominals.device == device
        assert simple_prior.prior_data.covariance_matrix.device == device
        assert simple_prior.nuisance_filter.device == device
        assert simple_prior._flipped_mask.device == device
        assert simple_prior.cyclical_mask.device == device

    def test_to_honours_its_argument(self, simple_prior):
        """`to()` must adopt the device it was given, not re-detect one."""
        simple_prior.to("cpu")
        assert simple_prior.device == torch.device("cpu")
        assert simple_prior.prior_data.nominals.device == torch.device("cpu")

    def test_to_accepts_a_torch_device(self, simple_prior):
        simple_prior.to(torch.device("cpu"))
        assert simple_prior.device == torch.device("cpu")

    def test_to_returns_self_for_chaining(self, simple_prior):
        assert simple_prior.to("cpu") is simple_prior

    def test_sampled_values_land_on_the_prior_device(self, simple_prior):
        assert simple_prior.sample((4,)).device == simple_prior.device


# ─────────────────────────────────────────────────────────────────────────────
# Caching
# ─────────────────────────────────────────────────────────────────────────────


class TestPriorDataCaching:
    def test_prior_data_is_cached(self, simple_prior):
        assert simple_prior.prior_data is simple_prior.prior_data

    def test_effective_bounds_are_cached(self, simple_prior):
        first = simple_prior.effective_lower_bounds
        assert simple_prior.effective_lower_bounds is first

    def test_to_invalidates_the_cache(self, simple_prior):
        before = simple_prior.prior_data
        simple_prior.to("cpu")
        assert simple_prior.prior_data is not before

    def test_cached_values_stay_correct(self, prior_data):
        """Caching must not change what the filtered view contains."""
        prior = Prior(prior_data, flat_msk=[True] * N_PARAMS)
        expected = prior._prior_data[prior.nuisance_filter]
        torch.testing.assert_close(prior.prior_data.nominals, expected.nominals)
        torch.testing.assert_close(prior.prior_data.lower_bounds, expected.lower_bounds)

    def test_check_bounds_stays_on_the_caller_device(self, simple_prior):
        params = torch.zeros(8, N_PARAMS)
        assert simple_prior.check_bounds(params).device == params.device

    def test_check_bounds_still_rejects_out_of_range(self, simple_prior):
        params = torch.stack([torch.zeros(N_PARAMS), torch.full((N_PARAMS,), 99.0)])
        torch.testing.assert_close(
            simple_prior.check_bounds(params),
            torch.tensor([True, False]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pickle round-trips
# ─────────────────────────────────────────────────────────────────────────────


class TestPriorPickling:
    def test_round_trip_preserves_values(self, simple_prior):
        restored = pickle.loads(pickle.dumps(simple_prior))
        torch.testing.assert_close(
            restored.prior_data.nominals, simple_prior.prior_data.nominals
        )
        assert restored.n_params == simple_prior.n_params

    def test_load_re_establishes_the_device(self, simple_prior):
        """Device state is re-derived on load, not carried over from the pickle."""
        restored = pickle.loads(pickle.dumps(simple_prior))
        assert restored.device == get_device()
        assert restored.prior_data.nominals.device == get_device()

    def test_stale_device_in_the_pickle_is_overridden(self, simple_prior):
        """A prior saved on another machine must not keep that machine's device."""
        payload = simple_prior.__getstate__()
        payload["_device"] = torch.device("meta")
        restored = Prior.__new__(Prior)
        restored.__setstate__(payload)
        assert restored.device == get_device()

    def test_legacy_pickle_without_device_loads(self, simple_prior):
        """Priors pickled before get_device() carried a handler object."""

        class _LegacyHandler:
            device = "definitely-not-a-device"

        payload = simple_prior.__getstate__()
        payload["device_handler"] = _LegacyHandler()
        restored = Prior.__new__(Prior)
        restored.__setstate__(payload)

        assert restored.device == get_device()
        assert not hasattr(restored, "device_handler")

    def test_caches_are_not_pickled(self, simple_prior):
        _ = simple_prior.prior_data  # populate
        state = simple_prior.__getstate__()
        assert "_prior_data_cache" not in state
        assert "_effective_bounds" not in state

    def test_restored_prior_is_usable(self, simple_prior):
        restored = pickle.loads(pickle.dumps(simple_prior))
        assert restored.sample((3,)).shape == (3, N_PARAMS)
        assert restored.check_bounds(torch.zeros(3, N_PARAMS)).all()


# ─────────────────────────────────────────────────────────────────────────────
# save / load_prior
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadPrior:
    def test_save_and_load(self, simple_prior, tmp_path):
        path = tmp_path / "prior.pkl"
        simple_prior.save(path)
        loaded = load_prior(path)
        assert loaded.n_params == simple_prior.n_params
        assert loaded.device == get_device()

    def test_load_honours_an_explicit_device(self, simple_prior, tmp_path):
        path = tmp_path / "prior.pkl"
        simple_prior.save(path)
        assert load_prior(path, device="cpu").device == torch.device("cpu")

    def test_missing_file_message_names_the_path(self, tmp_path):
        missing = tmp_path / "nope.pkl"
        with pytest.raises(PriorNotFound) as exc:
            load_prior(missing)
        assert str(missing) in str(exc.value)

    def test_wrong_contents_message_names_the_path(self, tmp_path):
        path = tmp_path / "notaprior.pkl"
        path.write_bytes(pickle.dumps({"not": "a prior"}))
        with pytest.raises(PriorNotFound) as exc:
            load_prior(path)
        assert str(path) in str(exc.value)


class TestLegacyDeviceHandlerPickle:
    """Priors saved before get_device() embedded a TorchDeviceHandler."""

    def test_legacy_handler_class_still_resolves(self):
        """Unpickling an old prior needs the class path to still exist."""
        from mach3sbitools.utils.device_handler import TorchDeviceHandler

        assert TorchDeviceHandler().device == str(get_device())

    def test_old_style_pickle_loads_and_is_usable(self, simple_prior):
        from mach3sbitools.utils.device_handler import TorchDeviceHandler

        state = simple_prior.__getstate__()
        state["device_handler"] = TorchDeviceHandler()

        cls, restored_state = pickle.loads(pickle.dumps((Prior, state)))
        restored = cls.__new__(cls)
        restored.__setstate__(restored_state)

        assert restored.device == get_device()
        assert not hasattr(restored, "device_handler")
        assert restored.sample((2,)).shape == (2, N_PARAMS)
