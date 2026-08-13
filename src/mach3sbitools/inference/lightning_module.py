"""
PyTorch Lightning module for SBI density estimator training.
"""

import time

import lightning as L
import torch
from sbi.neural_nets.estimators.base import ConditionalEstimator
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from torch.distributed._composable.fsdp import fully_shard


from mach3sbitools.data_processors import CompressorBase
from mach3sbitools.utils import PosteriorConfig, TrainingConfig

_EXPENSIVE_LOG_EVERY_N_EPOCHS = 10

# torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)


class SBILightningModule(L.LightningModule):
    """
    Lightning wrapper for an ``sbi`` density estimator.

    Handles the training and validation steps, EMA-smoothed validation loss
    tracking, and the learning rate scheduler.

    Metrics logged to TensorBoard
    ------------------------------
    Every epoch:
        train/loss, train/loss_std          — training loss mean and spread
        val/loss, val/loss_std              — validation loss mean and spread
        val/ema_loss                        — EMA-smoothed validation loss
        diagnostics/train_val_gap           — overfitting signal
        diagnostics/loss_improvement        — absolute epoch-on-epoch improvement
        diagnostics/relative_improvement    — scale-independent convergence signal
        diagnostics/ema_stability           — EMA bounce (high = LR too large)
        perf/samples_per_sec                — training throughput
        perf/epoch_time_sec                 — wall-clock epoch time
        perf/steps_per_epoch                — batches processed
        perf/effective_batch_size           — batch_size x world_size
        optim/lr_group_N                    — current learning rate(s)
        gpu/allocated_mb                    — VRAM allocated
        gpu/reserved_mb                     — VRAM reserved
        gpu/memory_pressure                 — fraction of total VRAM used

    Every ``_EXPENSIVE_LOG_EVERY_N_EPOCHS`` epochs:
        train/grad_norm                     — global gradient norm
        train/param_norm                    — global parameter norm
        grad_norms/<layer>                  — per-layer gradient norms
        weights/<layer>/std                 — per-layer weight standard deviation
        weights/<layer>/max_abs             — per-layer maximum absolute weight
    """

    def __init__(
        self,
        density_estimator: ConditionalEstimator,
        config: TrainingConfig,
        model_config: PosteriorConfig | None = None,
        x_compressor: CompressorBase | None = None,
        theta_compressor: CompressorBase | None = None,
    ):
        """
        :param density_estimator: The ``sbi`` density estimator to train.
        :param config: Training loop hyperparameters.
        :param model_config: Architecture config embedded in every checkpoint.
        """
        super().__init__()
        self.model = density_estimator
        self.config = config
        self.model_config = model_config
        # Needed for scheduling
        self.lr = config.learning_rate
        self.save_hyperparameters(ignore=["density_estimator"])

        # EMA state
        self.ema_val_loss: float = float("inf")

        # Diagnostics state
        self._prev_val_loss: float = float("inf")
        self._prev_ema_loss: float = float("inf")

        # Throughput state
        self._epoch_start_time: float = 0.0
        self._train_samples_seen: int = 0

        self._x_compressor = x_compressor
        self._theta_compressor = theta_compressor

    # ── Forward ───────────────────────────────────────────────────────────────
    def configure_model(self) -> None:
        """Apply FSDP2 sharding for ModelParallelStrategy."""
        if self.device_mesh is None:
            return  # not using ModelParallelStrategy — nothing to do

        # Shard leaf modules first, then the top-level model.
        # Adjust the module types below to match your density estimator's
        # actual submodules (e.g. its flow transform blocks / MLPs).
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Linear,)):
                fully_shard(module, mesh=self.device_mesh)

        fully_shard(self.model, mesh=self.device_mesh)

    def forward(self, theta: torch.Tensor, x: torch.Tensor):
        """
        Forward pass — delegates to the density estimator's loss method.

        :param theta: Parameter tensor of shape ``(batch_size, n_params)``.
        :param x: Observable tensor of shape ``(batch_size, n_bins)``.
        :returns: Per-sample loss tensor of shape ``(batch_size,)``.
        """
        return self.model.loss(theta, x)

    # ── Training ──────────────────────────────────────────────────────────────

    def on_train_epoch_start(self) -> None:
        """Record epoch start time and reset sample counter."""
        self._epoch_start_time = time.perf_counter()
        self._train_samples_seen = 0

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        Compute and log the training loss for one batch.

        :param batch: Tuple of ``(theta, x)`` tensors.
        :param batch_idx: Index of the current batch.
        :returns: Scalar mean loss.
        """
        theta, x = batch
        loss_per_sample = self.model.loss(theta, x)
        loss = loss_per_sample.mean()

        self._train_samples_seen += theta.shape[0]

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/loss_std",
            loss_per_sample.std(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        elapsed = time.perf_counter() - self._epoch_start_time
        do_expensive = self.current_epoch % _EXPENSIVE_LOG_EVERY_N_EPOCHS == 0

        # ── Throughput ────────────────────────────────────────────────────
        self.log(
            "perf/samples_per_sec",
            self._train_samples_seen / max(elapsed, 1e-6),
            sync_dist=True,
        )
        self.log("perf/epoch_time_sec", elapsed, sync_dist=True)
        self.log(
            "perf/steps_per_epoch",
            float(self._train_samples_seen) / self.config.batch_size,
            sync_dist=True,
        )
        self.log(
            "perf/effective_batch_size",
            float(self.config.batch_size * self.trainer.world_size),
            sync_dist=True,
        )

        # ── Learning rate ─────────────────────────────────────────────────
        for i, pg in enumerate(self.optimizers().param_groups):  # type: ignore
            self.log(f"optim/lr_group_{i}", pg["lr"], sync_dist=True)

        # ── GPU memory ────────────────────────────────────────────────────
        if torch.cuda.is_available() and self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            total = torch.cuda.get_device_properties(self.device).total_memory / 1024**2
            self.log("gpu/allocated_mb", allocated, sync_dist=True)
            self.log("gpu/reserved_mb", reserved, sync_dist=True)
            self.log("gpu/memory_pressure", allocated / total, sync_dist=True)

        # ── Expensive: weight statistics ──────────────────────────────────
        if do_expensive and self.trainer.is_global_zero:
            total_param_norm_sq = 0.0
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                # REMOVED logger_kwargs
                self.log(f"weights/{name}/std", p.data.std(), sync_dist=False)
                self.log(f"weights/{name}/max_abs", p.data.abs().max(), sync_dist=False)
                total_param_norm_sq += p.data.norm(2).item() ** 2
            self.log("train/param_norm", total_param_norm_sq**0.5, sync_dist=False)

    def on_before_optimizer_step(self, optimizer) -> None:
        if (
            self.current_epoch % _EXPENSIVE_LOG_EVERY_N_EPOCHS != 0
            or not self.trainer.is_global_zero
        ):
            return

        total_grad_norm_sq = 0.0
        for name, p in self.model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            layer_grad_norm = p.grad.data.norm(2).item()
            total_grad_norm_sq += layer_grad_norm**2
            # REMOVED logger_kwargs
            self.log(f"grad_norms/{name}", layer_grad_norm, sync_dist=False)

        self.log("train/grad_norm", total_grad_norm_sq**0.5, sync_dist=False)



    # ── Validation ────────────────────────────────────────────────────────────

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        Compute and log the validation loss for one batch.

        :param batch: Tuple of ``(theta, x)`` tensors.
        :param batch_idx: Index of the current batch.
        :returns: Scalar mean loss.
        """
        theta, x = batch
        loss_per_sample = self.model.loss(theta, x)
        loss = loss_per_sample.mean()

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val/loss_std",
            loss_per_sample.std(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        """
        Update EMA loss and log convergence diagnostics.
        """
        val_loss = float(self.trainer.callback_metrics.get("val/loss", float("inf")))

        # ── EMA update ────────────────────────────────────────────────────
        self.ema_val_loss = (
            val_loss
            if self.ema_val_loss == float("inf")
            else self.config.ema_alpha * val_loss
            + (1 - self.config.ema_alpha) * self.ema_val_loss
        )
        self.log("val/ema_loss", self.ema_val_loss, sync_dist=True, prog_bar=True)

        # ── EMA stability — large value means LR is too high ──────────────
        if self._prev_ema_loss != float("inf"):
            self.log(
                "diagnostics/ema_stability",
                abs(self.ema_val_loss - self._prev_ema_loss),
                sync_dist=True,
            )
        self._prev_ema_loss = self.ema_val_loss

        # ── Train / val gap — widening gap means overfitting ──────────────
        train_loss = self.trainer.callback_metrics.get("train/loss_epoch", float("inf"))
        if train_loss != float("inf"):
            self.log(
                "diagnostics/train_val_gap",
                val_loss - float(train_loss),
                sync_dist=True,
            )

        # ── Loss improvement ──────────────────────────────────────────────
        if self._prev_val_loss != float("inf"):
            improvement = self._prev_val_loss - val_loss
            self.log("diagnostics/loss_improvement", improvement, sync_dist=True)
            self.log(
                "diagnostics/relative_improvement",
                improvement / (abs(self._prev_val_loss) + 1e-8),
                sync_dist=True,
            )
        self._prev_val_loss = val_loss

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        """Adam + ReduceLROnPlateau scheduler monitoring ``val/ema_loss``."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.config.scheduler_patience,
            factor=0.5,
            min_lr=1e-8,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/ema_loss",
            },
        }

    # ── Checkpoint ────────────────────────────────────────────────────────────
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Embed model weights, architecture config, and epoch into checkpoint."""
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        checkpoint["model_state"] = get_model_state_dict(self.model, options=options)
        checkpoint["model_config"] = self.model_config
        checkpoint["epoch"] = self.current_epoch

        # Lets us load everything from a single checkpoint
        checkpoint["theta_dim"] = self.model.input_shape[0]
        checkpoint["theta_compressor"] = (
            self._theta_compressor.state_dict() if self._theta_compressor else None
        )

        checkpoint["x_dim"] = self.model.condition_shape[0]
        checkpoint["x_compressor"] = (
            self._x_compressor.state_dict() if self._x_compressor else None
        )
