"""
Tests for the PyTorch Lightning training components.

Covers:
  - inference/lightning_module.py  (SBILightningModule)
  - inference/lightning_datamodule.py  (SBIDataModule)
  - InferenceHandler.train_posterior — Lightning path
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import lightning as L
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from mach3sbitools.data_loaders import SBIDataModule
from mach3sbitools.data_processors import compressor_factory
from mach3sbitools.inference import InferenceHandler
from mach3sbitools.inference.lightning_module import SBILightningModule
from mach3sbitools.utils.config import TrainingConfig

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def DummyDataSet(n: int = 200, theta_dim: int = 4, x_dim: int = 6) -> TensorDataset:
    return TensorDataset(torch.randn(n, theta_dim), torch.randn(n, x_dim))


def _tiny_model(theta_dim: int = 4, x_dim: int = 6) -> torch.nn.Module:
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(x_dim, theta_dim)
            self.theta_dim = theta_dim
            self.x_dim = x_dim

        def forward(self, x):
            return self.fc(x)

        def loss(self, theta, x):
            return ((self.fc(x) - theta) ** 2).mean(dim=-1)

        @property
        def input_shape(self):
            return [self.theta_dim]

        @property
        def condition_shape(self):
            return [self.x_dim]

    return TinyModel()


def _minimal_config(tmp_path: Path, **kwargs) -> TrainingConfig:
    defaults: dict = dict(
        save_path=tmp_path / "model.ckpt",
        batch_size=32,
        max_epochs=2,
        stop_after_epochs=10,
        autosave_every=1,
        show_progress=False,
        validation_fraction=0.1,
        num_workers=0,
        ema_alpha=0.1,
        scheduler_patience=5,
    )
    defaults.update(kwargs)
    return TrainingConfig(**defaults)


def _fitted_compressor(n_features: int, n_components: int):
    """A PCA compressor fitted on throwaway data of the right width."""
    torch.manual_seed(0)
    return compressor_factory("pca", n_components=n_components).fit(
        torch.randn(200, n_features)
    )


@pytest.fixture()
def config(tmp_path) -> TrainingConfig:
    """A minimal TrainingConfig writing into this test's tmp_path."""
    return _minimal_config(tmp_path)


@pytest.fixture()
def module(config) -> SBILightningModule:
    """A module wrapping the tiny (4, 6) model, with logging suppressed."""
    module = SBILightningModule(_tiny_model(), config)
    with patch.object(module, "log"):
        yield module


@pytest.fixture()
def module_with_trainer(module) -> SBILightningModule:
    """
    A module with a mock trainer attached, for the epoch-end hooks.

    ``current_epoch`` reads through the trainer, and epoch 0 is an
    expensive-logging epoch, so the diagnostics paths are reachable.
    """
    module.trainer = MagicMock()
    module.trainer.world_size = 1
    module.trainer.current_epoch = 0
    module.trainer.is_global_zero = True
    return module


# ─────────────────────────────────────────────────────────────────────────────
# SBILightningModule
# ─────────────────────────────────────────────────────────────────────────────


class TestSBILightningModule:
    def test_forward_returns_per_sample_loss(self, module):
        """forward() should return a (batch_size,) loss tensor."""
        assert module(torch.randn(8, 4), torch.randn(8, 6)).shape == (8,)

    @pytest.mark.parametrize("step", ["training_step", "validation_step"])
    def test_step_returns_a_scalar_loss(self, module, step):
        loss = getattr(module, step)((torch.randn(8, 4), torch.randn(8, 6)), 0)
        assert loss.ndim == 0

    def test_configure_optimizers_returns_adam_with_scheduler(self, module):
        result = module.configure_optimizers()
        assert isinstance(result["optimizer"], torch.optim.Adam)
        assert result["lr_scheduler"]["monitor"] == "val/loss"


class TestEmaValidationLoss:
    """EMA update: ``alpha * new + (1 - alpha) * old``, seeded on first use."""

    def _end_epoch(self, module, val_loss: float):
        module.trainer.callback_metrics = {"val/loss": torch.tensor(val_loss)}
        with patch.object(module, "log") as logged:
            module.on_validation_epoch_end()
        return logged

    def test_first_epoch_seeds_from_val_loss(self, module_with_trainer):
        self._end_epoch(module_with_trainer, 2.0)
        assert module_with_trainer.ema_val_loss == pytest.approx(2.0)

    def test_subsequent_epoch_is_a_weighted_average(self, module_with_trainer):
        module_with_trainer.ema_val_loss = 2.0
        self._end_epoch(module_with_trainer, 1.0)
        assert module_with_trainer.ema_val_loss == pytest.approx(0.1 * 1.0 + 0.9 * 2.0)

    def test_ema_is_logged_for_early_stopping_to_monitor(self, module_with_trainer):
        logged = self._end_epoch(module_with_trainer, 1.5)
        assert "val/ema_loss" in [call[0][0] for call in logged.call_args_list]


class TestEpochEndLogging:
    def _logged_keys(self, module) -> list[str]:
        with patch.object(module, "log") as logged:
            module.on_train_epoch_end()
        return [call[0][0] for call in logged.call_args_list]

    def test_gpu_metrics_skipped_without_cuda(self, module_with_trainer):
        with patch("torch.cuda.is_available", return_value=False):
            keys = self._logged_keys(module_with_trainer)
        assert not any(k.startswith("gpu/") for k in keys)

    def test_throughput_metrics_are_logged(self, module_with_trainer):
        keys = self._logged_keys(module_with_trainer)
        assert "perf/samples_per_sec" in keys
        assert "perf/epoch_time_sec" in keys

    def test_parameter_norms_logged_once_per_expensive_epoch(self, module_with_trainer):
        """Norms must be reported as one fused stack, not one sync per layer."""
        keys = self._logged_keys(module_with_trainer)
        assert "train/param_norm" in keys
        assert sum(k.startswith("weights/") for k in keys) == len(
            list(module_with_trainer.model.parameters())
        )


class TestGradientNormLogging:
    """Gradient norms are a per-epoch health check, not a per-step metric."""

    def _step(self, module) -> list[str]:
        with patch.object(module, "log") as logged:
            module.on_before_optimizer_step(MagicMock())
        return [call[0][0] for call in logged.call_args_list]

    def _with_grads(self, module):
        """Populate .grad so there is something to take a norm of."""
        module.model.loss(torch.randn(4, 4), torch.randn(4, 6)).mean().backward()
        return module

    def test_logs_on_the_first_step_of_an_expensive_epoch(self, module_with_trainer):
        module = self._with_grads(module_with_trainer)
        module.on_train_epoch_start()
        assert "train/grad_norm" in self._step(module)

    def test_does_not_log_again_within_the_same_epoch(self, module_with_trainer):
        module = self._with_grads(module_with_trainer)
        module.on_train_epoch_start()
        self._step(module)
        assert self._step(module) == []

    def test_logs_again_after_the_next_epoch_starts(self, module_with_trainer):
        module = self._with_grads(module_with_trainer)
        module.on_train_epoch_start()
        self._step(module)
        module.on_train_epoch_start()
        assert "train/grad_norm" in self._step(module)


class TestSBILightningModuleCompression:
    """
    Compressors handed to the module must actually reach the batches.

    Storing them for the checkpoint but training on raw data leaves the model
    in a different space from the one it is later conditioned in.
    """

    def test_compress_batch_is_identity_without_compressors(self, tmp_path):
        module = SBILightningModule(_tiny_model(), _minimal_config(tmp_path))
        theta, x = torch.randn(8, 4), torch.randn(8, 6)
        out_theta, out_x = module.compress_batch(theta, x)
        torch.testing.assert_close(out_theta, theta)
        torch.testing.assert_close(out_x, x)

    def test_compress_batch_reduces_both_dimensions(self, tmp_path):
        module = SBILightningModule(
            _tiny_model(theta_dim=2, x_dim=3),
            _minimal_config(tmp_path),
            x_compressor=_fitted_compressor(6, 3),
            theta_compressor=_fitted_compressor(4, 2),
        )
        theta, x = module.compress_batch(torch.randn(8, 4), torch.randn(8, 6))
        assert theta.shape == (8, 2)
        assert x.shape == (8, 3)

    def test_compress_batch_applies_only_the_supplied_compressor(self, tmp_path):
        module = SBILightningModule(
            _tiny_model(theta_dim=4, x_dim=3),
            _minimal_config(tmp_path),
            x_compressor=_fitted_compressor(6, 3),
        )
        theta_in = torch.randn(8, 4)
        theta, x = module.compress_batch(theta_in, torch.randn(8, 6))
        torch.testing.assert_close(theta, theta_in)
        assert x.shape == (8, 3)

    def test_training_step_runs_on_compressed_batch(self, tmp_path):
        """The model is sized for compressed input; a raw batch would not fit."""
        module = SBILightningModule(
            _tiny_model(theta_dim=2, x_dim=3),
            _minimal_config(tmp_path),
            x_compressor=_fitted_compressor(6, 3),
            theta_compressor=_fitted_compressor(4, 2),
        )
        with patch.object(module, "log"):
            loss = module.training_step((torch.randn(8, 4), torch.randn(8, 6)), 0)
        assert loss.ndim == 0

    def test_compressors_are_written_to_the_checkpoint(self, tmp_path):
        module = SBILightningModule(
            _tiny_model(theta_dim=2, x_dim=3),
            _minimal_config(tmp_path),
            x_compressor=_fitted_compressor(6, 3),
            theta_compressor=_fitted_compressor(4, 2),
        )
        checkpoint: dict = {}
        module.on_save_checkpoint(checkpoint)
        assert checkpoint["x_compressor"]["type"] == "pca"
        assert checkpoint["theta_compressor"]["type"] == "pca"


# ─────────────────────────────────────────────────────────────────────────────
# SBIDataModule
# ─────────────────────────────────────────────────────────────────────────────


class TestSBIDataModule:
    def test_setup_splits_dataset_correctly(self, tmp_path):
        """setup() should produce train/val splits summing to the full dataset."""
        n = 200
        cfg = _minimal_config(tmp_path)
        cfg.validation_fraction = 0.1
        dm = SBIDataModule(DummyDataSet(n=n), cfg)
        assert dm.train_dataset is None  # before setup
        dm.setup()
        assert len(dm.train_dataset) + len(dm.val_dataset) == n
        assert len(dm.val_dataset) == pytest.approx(20, abs=1)

    def test_setup_is_deterministic(self, tmp_path):
        """All DDP ranks must derive the same split from the same seed."""
        ds = DummyDataSet(n=200)
        cfg = _minimal_config(tmp_path)
        dm1 = SBIDataModule(ds, cfg)
        dm2 = SBIDataModule(ds, cfg)
        dm1.setup()
        dm2.setup()
        assert dm1.train_dataset.indices == dm2.train_dataset.indices
        assert dm1.val_dataset.indices == dm2.val_dataset.indices

    def test_split_is_a_permutation_not_a_contiguous_slice(self, tmp_path):
        """
        Merged shards arrive grouped by source file, so a contiguous tail
        would make the validation set unrepresentative.
        """
        n = 200
        cfg = _minimal_config(tmp_path)
        dm = SBIDataModule(DummyDataSet(n=n), cfg)
        dm.setup()

        train_idx = list(dm.train_dataset.indices)
        val_idx = list(dm.val_dataset.indices)

        assert sorted(train_idx + val_idx) == list(range(n))
        assert train_idx != sorted(train_idx)
        assert val_idx != list(range(len(train_idx), n))

    def test_setup_accepts_stage_argument(self, tmp_path):
        dm = SBIDataModule(DummyDataSet(), _minimal_config(tmp_path))
        dm.setup(stage="fit")  # must not raise
        assert dm.train_dataset is not None

    def test_train_dataloader_properties(self, tmp_path):
        """Train loader should shuffle, drop last, and use the right batch size."""
        cfg = _minimal_config(tmp_path)
        cfg.batch_size = 16
        dm = SBIDataModule(DummyDataSet(n=200), cfg)
        dm.setup()
        loader = dm.train_dataloader()
        assert isinstance(loader, DataLoader)
        assert isinstance(loader.sampler, RandomSampler)
        assert loader.drop_last is True
        assert loader.batch_size == 16
        assert loader.num_workers == 0

    def test_val_dataloader_properties(self, tmp_path):
        """Val loader should not shuffle and use zero workers."""
        dm = SBIDataModule(DummyDataSet(), _minimal_config(tmp_path))
        dm.setup()
        loader = dm.val_dataloader()
        assert isinstance(loader, DataLoader)
        assert isinstance(loader.sampler, SequentialSampler)
        assert loader.num_workers == 0


# ─────────────────────────────────────────────────────────────────────────────
# Integration: end-to-end Lightning training
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestLightningTrainingIntegration:
    def _run_trainer(self, tmp_path, **cfg_kwargs):
        cfg = _minimal_config(tmp_path, **cfg_kwargs)
        module = SBILightningModule(_tiny_model(), cfg)
        dm = SBIDataModule(DummyDataSet(n=200), cfg)
        trainer = L.Trainer(
            max_epochs=cfg.max_epochs,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(module, datamodule=dm)
        return module, trainer

    def test_trainer_fits_and_updates_ema(self, tmp_path):
        """Full fit() should succeed and update EMA away from inf."""
        module, _ = self._run_trainer(tmp_path, max_epochs=3, stop_after_epochs=50)
        assert module.ema_val_loss != float("inf")

    def test_checkpoint_written(self, tmp_path):
        cfg = _minimal_config(tmp_path, max_epochs=2, autosave_every=1)
        module = SBILightningModule(_tiny_model(), cfg)
        dm = SBIDataModule(DummyDataSet(n=200), cfg)
        ckpt_cb = ModelCheckpoint(
            dirpath=tmp_path / "checkpoints",
            monitor="val/ema_loss",
            save_top_k=1,
            every_n_epochs=1,
        )
        trainer = L.Trainer(
            max_epochs=2,
            callbacks=[ckpt_cb],
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            logger=False,
        )
        trainer.fit(module, datamodule=dm)
        assert len(list((tmp_path / "checkpoints").glob("*.ckpt"))) > 0


# ─────────────────────────────────────────────────────────────────────────────
# InferenceHandler.train_posterior — Lightning path
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestInferenceHandlerLightning:
    def test_train_posterior_guard_conditions(
        self, prior_save, merged_data_dir, training_config, posterior_config
    ):
        """Missing data or inference object should raise a helpful ValueError."""
        # No dataset
        handler = InferenceHandler(prior_save)
        handler.create_posterior(posterior_config)
        with pytest.raises(ValueError, match="set_dataset"):
            handler.train_posterior(training_config)

        # No inference object
        handler2 = InferenceHandler(prior_save)
        handler2.set_dataset(merged_data_dir)
        with pytest.raises(ValueError, match="create_posterior"):
            handler2.train_posterior(training_config)

    def test_train_posterior_sets_density_estimator_in_eval_mode(
        self, prior_save, merged_data_dir, posterior_config, tmp_path
    ):
        cfg = TrainingConfig(
            save_path=tmp_path / "model.ckpt",
            max_epochs=2,
            stop_after_epochs=50,
            batch_size=256,
            show_progress=False,
            autosave_every=500,
            num_workers=0,
        )
        handler = InferenceHandler(prior_save)
        handler.set_dataset(merged_data_dir)
        handler.create_posterior(posterior_config)
        handler.train_posterior(cfg, model_config=posterior_config)

        assert handler._density_estimator is not None
        assert not handler._density_estimator.training

    def test_slurm_nnodes_env_respected(
        self, prior_save, merged_data_dir, posterior_config, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("SLURM_NNODES", "1")
        handler = InferenceHandler(prior_save)
        handler.set_dataset(merged_data_dir)
        handler.create_posterior(posterior_config)

        cfg = TrainingConfig(
            save_path=tmp_path / "model.ckpt",
            max_epochs=1,
            batch_size=256,
            show_progress=False,
            autosave_every=500,
            num_workers=0,
        )
        captured = {}
        original_init = L.Trainer.__init__

        def patched_init(self, *args, **kwargs):
            captured["num_nodes"] = kwargs.get("num_nodes")
            kwargs.update(accelerator="cpu", devices=1, strategy="auto")
            original_init(self, *args, **kwargs)

        with patch.object(L.Trainer, "__init__", patched_init):
            handler.train_posterior(cfg)

        assert captured["num_nodes"] == 1
