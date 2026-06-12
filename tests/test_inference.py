"""
Tests for mach3sbitools.inference.InferenceHandler.

Covers error guards, the full training → sampling lifecycle, and checkpoint
round-trips. The session-scoped trained_handler fixture is shared across all
happy-path and checkpoint tests so training only runs once.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from scipy import stats
from torch.utils.data import TensorDataset

from mach3sbitools.inference import InferenceHandler
from mach3sbitools.simulator import load_prior
from mach3sbitools.utils import TorchDeviceHandler

N_SAMPLES = 10_000


# ─────────────────────────────────────────────────────────────────────────────
# Session fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def nominal_observation(simulator_injector):
    rng = np.random.default_rng(42)
    dh = TorchDeviceHandler()
    return dh.to_tensor(
        rng.poisson(lam=1, size=len(simulator_injector.get_data_bins()))
        .astype(np.float32)
        .tolist()
    )


@pytest.fixture(scope="session")
def trained_handler(prior_save, dummy_data_dir, posterior_config, training_config):
    handler = InferenceHandler(prior_save)
    handler.set_dataset(dummy_data_dir)
    handler.load_training_data()
    handler.create_posterior(posterior_config)
    handler.train_posterior(training_config)
    return handler


@pytest.fixture(scope="session")
def samples(trained_handler, nominal_observation):
    return trained_handler.sample_posterior(N_SAMPLES, nominal_observation)


# ─────────────────────────────────────────────────────────────────────────────
# Guard / error paths
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestInferenceHandlerErrorPaths:
    def test_load_training_data_requires_dataset(self, prior_save):
        with pytest.raises(ValueError):
            InferenceHandler(prior_save).load_training_data()

    def test_train_posterior_requires_tensor_dataset(
        self, prior_save, posterior_config, training_config
    ):
        handler = InferenceHandler(prior_save)
        handler.create_posterior(posterior_config)
        with pytest.raises(ValueError, match="load_training_data"):
            handler.train_posterior(training_config)

    def test_train_posterior_requires_inference(self, prior_save, training_config):
        handler = InferenceHandler(prior_save)
        handler._tensor_dataset = TensorDataset(torch.zeros(10, 4), torch.zeros(10, 6))
        with pytest.raises(ValueError, match="create_posterior"):
            handler.train_posterior(training_config)

    def test_build_posterior_requires_density_estimator_and_inference(self, prior_save):
        handler = InferenceHandler(prior_save)
        with pytest.raises(ValueError, match="density estimator"):
            handler.build_posterior()

        handler._density_estimator = MagicMock()
        with pytest.raises(ValueError):
            handler.build_posterior()

    def test_sample_and_log_likelihood_require_posterior(self, prior_save):
        handler = InferenceHandler(prior_save)
        handler._density_estimator = MagicMock()
        handler.inference = MagicMock()
        handler.posterior = None
        with patch.object(handler, "build_posterior"):
            with pytest.raises(ValueError, match="density estimator"):
                handler.sample_posterior(10, [1.0] * 12)
            with pytest.raises(ValueError, match="density estimator"):
                handler.get_log_likelihood(np.ones((5, 4)), [1.0] * 12)

    def test_load_posterior_file_not_found(self, prior_save):
        with pytest.raises(FileNotFoundError):
            InferenceHandler(prior_save).load_posterior(Path("/no/such/checkpoint.pt"))


# ─────────────────────────────────────────────────────────────────────────────
# Happy path — training lifecycle
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestInferenceHandlerHappyPath:
    def test_full_lifecycle_setup(self, prior_save, dummy_data_dir, posterior_config):
        """Each lifecycle step should leave the handler in the expected state."""
        handler = InferenceHandler(prior_save)
        handler.set_dataset(dummy_data_dir)
        assert handler.dataset is not None
        handler.load_training_data()
        assert handler._tensor_dataset is not None
        handler.create_posterior(posterior_config)
        assert handler.inference is not None

    def test_density_estimator_in_eval_mode_after_training(self, trained_handler):
        assert trained_handler._density_estimator is not None
        assert not trained_handler._density_estimator.training

    def test_posterior_samples_shape_and_validity(self, samples, trained_handler):
        """Samples should have the right shape, be finite, and lie within prior bounds."""
        n_params = trained_handler.prior.n_params
        assert samples.shape == (N_SAMPLES, n_params)
        assert torch.all(torch.isfinite(samples))

        s = samples.cpu()
        lower = trained_handler.prior.prior_data.lower_bounds.cpu()
        upper = trained_handler.prior.prior_data.upper_bounds.cpu()
        assert torch.all(s >= lower) and torch.all(s <= upper)

    def test_posterior_is_not_wildly_off_prior(self, samples, trained_handler):
        """Posterior mean should stay within 3σ of the prior nominal."""
        nominals = trained_handler.prior.prior_data.nominals.cpu()
        prior_std = trained_handler.prior.variance.cpu().sqrt()
        deviation = (samples.cpu().mean(dim=0) - nominals).abs()
        assert torch.all(deviation < 3 * prior_std)

    def test_sampling_is_reproducible(self, trained_handler, nominal_observation):
        torch.manual_seed(42)
        a = trained_handler.sample_posterior(100, nominal_observation)
        torch.manual_seed(42)
        b = trained_handler.sample_posterior(100, nominal_observation)
        assert torch.allclose(a, b)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint round-trips
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestInferenceHandlerCheckpoints:
    def test_save_and_reload_produces_equivalent_posterior(
        self,
        trained_handler,
        tmp_path,
        prior_save,
        posterior_config,
        nominal_observation,
    ):
        """State-dict round-trip should yield statistically indistinguishable samples."""
        ckpt_path = tmp_path / "de.pt"
        torch.save(trained_handler._density_estimator.state_dict(), ckpt_path)

        theta_dim = trained_handler._tensor_dataset.tensors[0].shape[1]
        x_dim = trained_handler._tensor_dataset.tensors[1].shape[1]

        loaded = InferenceHandler(prior_save)
        loaded.create_posterior(posterior_config)
        device = loaded.device_handler.device
        de = loaded.inference._build_neural_net(
            torch.zeros(2, theta_dim, device=device),
            torch.zeros(2, x_dim, device=device),
        )
        de.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        de.to(device).eval()
        loaded._density_estimator = de

        original = (
            trained_handler.sample_posterior(500, nominal_observation).cpu().numpy()
        )
        reloaded = loaded.sample_posterior(500, nominal_observation).cpu().numpy()

        for i in range(original.shape[1]):
            ks_stat, _ = stats.ks_2samp(original[:, i], reloaded[:, i], method="asymp")
            assert ks_stat < 0.1, f"Parameter {i}: KS={ks_stat:.3f}"

    def test_load_posterior_from_autosave_format(
        self, prior_save, posterior_config, tmp_path
    ):
        prior = load_prior(prior_save)
        neural_net = posterior_nn(
            model=posterior_config.model,
            hidden_features=posterior_config.hidden_features,
            num_transforms=posterior_config.num_transforms,
            dropout_probability=posterior_config.dropout_probability,
            num_blocks=posterior_config.num_blocks,
            num_bins=posterior_config.num_bins,
            device=prior.device_handler.device,
            z_score_x="structured",
            z_score_theta="independent",
        )
        npe = NPE(
            prior=prior,
            density_estimator=neural_net,
            device=prior.device_handler.device,
        )
        theta_dim = len(prior.prior_data.parameter_names)
        de = npe._build_neural_net(torch.zeros(2, theta_dim), torch.zeros(2, 12))

        ckpt_path = tmp_path / "autosave.ckpt"
        torch.save(
            {
                "epoch": 1,
                "model_state": de.state_dict(),
                "model_config": posterior_config,
                "theta_dim": theta_dim,
                "x_dim": 12,
                "x_compressor": None,
                "theta_compressor": None,
            },
            ckpt_path,
        )

        handler = InferenceHandler(prior_save)
        handler.load_posterior(ckpt_path)
        assert handler._density_estimator is not None
