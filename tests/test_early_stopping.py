"""
Tests for what training monitors and when it stops.

A smoothed metric lags the true optimum, so early stopping, checkpoint
selection and the LR schedule all watch the raw validation loss. The EMA is
kept as a logged diagnostic only.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from mach3sbitools.inference import InferenceHandler
from mach3sbitools.inference.lightning_module import SBILightningModule
from mach3sbitools.utils import TrainingConfig


@pytest.fixture()
def callbacks(prior_save, make_training_config):
    """The callback stack the handler builds for a given config."""

    def _build(**overrides):
        handler = InferenceHandler(prior_save)
        return handler._build_callbacks(make_training_config(**overrides))

    return _build


def _only(callbacks, kind):
    return next(c for c in callbacks if isinstance(c, kind))


# ─────────────────────────────────────────────────────────────────────────────
# What is monitored
# ─────────────────────────────────────────────────────────────────────────────


class TestMonitoredMetric:
    """
    Nothing may decide on ``val/ema_loss``.

    Its minimum trails the raw loss by roughly ``(1 - alpha) / alpha`` epochs,
    which delays stopping and makes ModelCheckpoint keep a later, worse epoch.
    """

    def test_early_stopping_watches_raw_val_loss(self, callbacks):
        assert _only(callbacks(), EarlyStopping).monitor == "val/loss"

    def test_checkpointing_watches_raw_val_loss(self, callbacks):
        assert _only(callbacks(), ModelCheckpoint).monitor == "val/loss"

    def test_scheduler_watches_raw_val_loss(self, tmp_path):
        module = SBILightningModule(
            torch.nn.Linear(2, 2), TrainingConfig(save_path=tmp_path / "m.ckpt")
        )
        assert module.configure_optimizers()["lr_scheduler"]["monitor"] == "val/loss"

    def test_ema_is_still_logged_as_a_diagnostic(self, tmp_path):
        """It stays available to look at, it just decides nothing."""
        module = SBILightningModule(
            torch.nn.Linear(2, 2), TrainingConfig(save_path=tmp_path / "m.ckpt")
        )
        module.trainer = MagicMock()
        module.trainer.callback_metrics = {"val/loss": torch.tensor(1.5)}

        with patch.object(module, "log") as logged:
            module.on_validation_epoch_end()

        assert "val/ema_loss" in [call[0][0] for call in logged.call_args_list]


# ─────────────────────────────────────────────────────────────────────────────
# min_delta
# ─────────────────────────────────────────────────────────────────────────────


class TestMinDelta:
    def test_defaults_to_zero(self, callbacks):
        assert _only(callbacks(), EarlyStopping).min_delta == 0.0

    def test_is_forwarded_from_the_config(self, callbacks):
        # Lightning stores it signed by mode; "min" keeps it as -min_delta.
        assert abs(_only(callbacks(min_delta=0.01), EarlyStopping).min_delta) == 0.01

    def test_patience_is_forwarded_from_the_config(self, callbacks):
        assert _only(callbacks(stop_after_epochs=7), EarlyStopping).patience == 7

    def test_noise_below_min_delta_does_not_reset_patience(self):
        """
        The point of min_delta: an improvement smaller than it must not count.

        Driven through the real callback rather than reimplementing its rule.
        """
        stopper = EarlyStopping(
            monitor="val/loss", patience=2, min_delta=0.01, mode="min"
        )
        stopper.best_score = torch.tensor(1.0)
        stopper.wait_count = 0

        stopper._evaluate_stopping_criteria(torch.tensor(0.999))
        assert stopper.wait_count == 1, "a 0.001 improvement should not count"

        stopper.wait_count = 0
        stopper._evaluate_stopping_criteria(torch.tensor(0.98))
        assert stopper.wait_count == 0, "a 0.02 improvement should count"


# ─────────────────────────────────────────────────────────────────────────────
# Batch limits
# ─────────────────────────────────────────────────────────────────────────────


class TestBatchLimits:
    @pytest.fixture()
    def trainer(self, prior_save, make_training_config):
        def _build(**overrides):
            handler = InferenceHandler(prior_save)
            return handler._build_trainer(make_training_config(**overrides))

        return _build

    def test_default_to_the_whole_split(self, trainer):
        built = trainer()
        assert built.limit_train_batches == 1.0
        assert built.limit_val_batches == 1.0

    def test_train_limit_is_forwarded(self, trainer):
        assert trainer(limit_train_batches=25).limit_train_batches == 25

    def test_val_limit_is_forwarded(self, trainer):
        assert trainer(limit_val_batches=5).limit_val_batches == 5

    @pytest.mark.slow
    def test_train_limit_caps_the_steps_actually_run(
        self, prior_save, merged_data_dir, posterior_config, make_training_config
    ):
        """A capped epoch must really be that many batches, not the full split."""
        limit = 3
        handler = InferenceHandler(prior_save)
        handler.set_dataset(merged_data_dir)
        handler.create_posterior(posterior_config)

        config = make_training_config(
            max_epochs=1, batch_size=64, limit_train_batches=limit
        )
        handler.train_posterior(config, model_config=posterior_config)

        uncapped = len(handler.dataset) * (1 - config.validation_fraction) // 64
        assert limit < uncapped


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────


def test_num_workers_default_overlaps_io_with_compute():
    """A single worker cannot keep a GPU fed from a disk-backed dataset."""
    assert TrainingConfig().num_workers >= 4
