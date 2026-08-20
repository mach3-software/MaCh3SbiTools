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
        print_interval=1,
        show_progress=False,
        validation_fraction=0.1,
        num_workers=0,
        ema_alpha=0.1,
        scheduler_patience=5,
    )
    defaults.update(kwargs)
    return TrainingConfig(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# SBILightningModule
# ─────────────────────────────────────────────────────────────────────────────


class TestSBILightningModule:
    def test_forward_returns_per_sample_loss(self, tmp_path):
        """forward() should return a (batch_size,) loss tensor."""
        module = SBILightningModule(_tiny_model(), _minimal_config(tmp_path))
        out = module(torch.randn(8, 4), torch.randn(8, 6))
        assert out.shape == (8,)

    def test_training_step_returns_scalar(self, tmp_path):
        module = SBILightningModule(_tiny_model(), _minimal_config(tmp_path))
        with patch.object(module, "log"):
            loss = module.training_step((torch.randn(8, 4), torch.randn(8, 6)), 0)
        assert loss.ndim == 0

    def test_validation_step_returns_scalar(self, tmp_path):
        module = SBILightningModule(_tiny_model(), _minimal_config(tmp_path))
        with patch.object(module, "log"):
            loss = module.validation_step((torch.randn(8, 4), torch.randn(8, 6)), 0)
        assert loss.ndim == 0

    def test_ema_first_epoch_equals_val_loss(self, tmp_path):
        """On the first epoch (ema=inf), EMA should collapse to val_loss exactly."""
        module = SBILightningModule(_tiny_model(), _minimal_config(tmp_path))
        module.trainer = MagicMock()
        module.trainer.callback_metrics = {"val/loss": torch.tensor(2.0)}
        with patch.object(module, "log"):
            module.on_validation_epoch_end()
        assert module.ema_val_loss == pytest.approx(2.0)

    def test_ema_subsequent_epoch_is_weighted_average(self, tmp_path):
        """EMA update: alpha*new + (1-alpha)*old."""
        cfg = _minimal_config(tmp_path)
        cfg.ema_alpha = 0.1
        module = SBILightningModule(_tiny_model(), cfg)
        module.ema_val_loss = 2.0
        module.trainer = MagicMock()
        module.trainer.callback_metrics = {"val/loss": torch.tensor(1.0)}
        with patch.object(module, "log"):
            module.on_validation_epoch_end()
        assert module.ema_val_loss == pytest.approx(0.1 * 1.0 + 0.9 * 2.0)

    def test_ema_logged_under_correct_key(self, tmp_path):
        module = SBILightningModule(_tiny_model(), _minimal_config(tmp_path))
        module.trainer = MagicMock()
        module.trainer.callback_metrics = {"val/loss": torch.tensor(1.5)}
        with patch.object(module, "log") as mock_log:
            module.on_validation_epoch_end()
        logged_keys = [c[0][0] for c in mock_log.call_args_list]
        assert "val/ema_loss" in logged_keys

    def test_gpu_logging_skipped_without_cuda(self, tmp_path):
        module = SBILightningModule(_tiny_model(), _minimal_config(tmp_path))
        module.trainer = MagicMock()
        module.trainer.world_size = 1
        with patch("torch.cuda.is_available", return_value=False):
            with patch.object(module, "log") as mock_log:
                module.on_train_epoch_end()
        logged_keys = [c[0][0] for c in mock_log.call_args_list]
        assert not any(k.startswith("gpu/") for k in logged_keys)

    def test_configure_optimizers_returns_adam_with_scheduler(self, tmp_path):
        module = SBILightningModule(_tiny_model(), _minimal_config(tmp_path))
        result = module.configure_optimizers()
        assert isinstance(result["optimizer"], torch.optim.Adam)
        assert "lr_scheduler" in result


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
        ds = DummyDataSet(n=200)
        cfg = _minimal_config(tmp_path)
        dm1 = SBIDataModule(ds, cfg)
        dm2 = SBIDataModule(ds, cfg)
        dm1.setup()
        dm2.setup()
        assert dm1.train_dataset.indices[:10] == dm2.train_dataset.indices[:10]

    def test_setup_accepts_stage_argument(self, tmp_path):
        dm = SBIDataModule(DummyDataSet(), _minimal_config(tmp_path))
        dm.setup(stage="fit")  # must not raise
        assert dm.train_dataset is not None

    def test_train_dataloader_properties(self, tmp_path):
        """Train loader should shuffle, drop last, use correct batch size."""
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
        self, prior_save, training_config, posterior_config
    ):
        """Missing data or inference object should raise ValueError with helpful message."""
        # No tensor dataset
        handler = InferenceHandler(prior_save)
        handler.create_posterior(posterior_config)
        with pytest.raises(ValueError, match="load_training_data"):
            handler.train_posterior(training_config)

        # No inference object
        handler2 = InferenceHandler(prior_save)
        handler2._training_dataset = TensorDataset(
            torch.zeros(10, 4), torch.zeros(10, 6)
        )
        with pytest.raises(ValueError, match="create_posterior"):
            handler2.train_posterior(training_config)

    def test_train_posterior_sets_density_estimator_in_eval_mode(
        self, prior_save, dummy_data_dir, posterior_config, tmp_path
    ):
        cfg = TrainingConfig(
            save_path=tmp_path / "model.ckpt",
            max_epochs=2,
            stop_after_epochs=50,
            batch_size=256,
            show_progress=False,
            autosave_every=500,
        )
        handler = InferenceHandler(prior_save)
        handler.set_dataset(dummy_data_dir)
        handler.load_training_data()
        handler.create_posterior(posterior_config)
        handler.train_posterior(cfg, model_config=posterior_config)

        assert handler._density_estimator is not None
        assert not handler._density_estimator.training

    def test_slurm_nnodes_env_respected(
        self, prior_save, dummy_data_dir, posterior_config, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("SLURM_NNODES", "1")
        handler = InferenceHandler(prior_save)
        handler.set_dataset(dummy_data_dir)
        handler.load_training_data()
        handler.create_posterior(posterior_config)

        cfg = TrainingConfig(
            save_path=tmp_path / "model.ckpt",
            max_epochs=1,
            batch_size=256,
            show_progress=False,
            autosave_every=500,
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
