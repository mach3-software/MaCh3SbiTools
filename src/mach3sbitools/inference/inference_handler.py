"""
High-level inference interface.

Lifecycle methods
-----------------
**Fresh training**::

    handler = InferenceHandler(prior_path)
    handler.set_dataset(data_folder)
    handler.load_training_data()          # loads into shared CPU RAM once
    handler.create_posterior(config)
    handler.train_posterior(training_config, model_config=config)

**Resume training from a checkpoint**::

    handler = InferenceHandler(prior_path)
    handler.set_dataset(data_folder)
    handler.load_training_data()
    handler.resume_training(checkpoint_path, training_config)
    # That's it — architecture is read from the checkpoint automatically.

**Inference only**::

    handler = InferenceHandler(prior_path)
    handler.load_posterior(checkpoint_path)
    samples = handler.sample_posterior(10_000, x_observed)

Dataset sharing
---------------
``load_training_data`` builds a single :class:`~torch.utils.data.TensorDataset`
in CPU RAM. Lightning's ``DistributedSampler`` then slices disjoint subsets per
GPU rank at training time — the tensor is never replicated across processes.
"""

from __future__ import annotations

import os
import pathlib
from pathlib import Path
from typing import Any, cast

import lightning
import numpy as np
import torch
import torch.autograd.graph
import torch.nn as nn
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import ModelParallelStrategy
from sbi.inference import NPE, DirectPosterior
from sbi.inference.posteriors.posterior_parameters import DirectPosteriorParameters
from sbi.neural_nets import posterior_nn
from torch.utils.data import TensorDataset

from mach3sbitools.data_loaders import SBIDataModule
from mach3sbitools.data_loaders.paraket_dataloader import ParaketDataset
from mach3sbitools.inference.lightning_module import SBILightningModule
from mach3sbitools.simulator import load_prior
from mach3sbitools.types import SimulatorData
from mach3sbitools.utils.config import PosteriorConfig, TrainingConfig
from mach3sbitools.utils.device_handler import TorchDeviceHandler
from mach3sbitools.utils.logger import get_logger

logger = get_logger()
torch.set_float32_matmul_precision("medium")
torch.serialization.add_safe_globals(
    [
        TrainingConfig,
        PosteriorConfig,
        pathlib.PosixPath,
        pathlib.WindowsPath,
        pathlib.Path,
    ]
)


def _select_accelerator_and_strategy(
    use_model_parallel: bool = False,
) -> tuple[str, str | ModelParallelStrategy]:
    if torch.cuda.is_available():
        return "gpu", ModelParallelStrategy() if use_model_parallel else "ddp"
    if torch.backends.mps.is_available():
        return "mps", "auto"
    return "cpu", "auto"


# ---------------------------------------------------------------------------
# Architecture kwargs filtering
# ---------------------------------------------------------------------------

# Arguments accepted by each model family. Any key absent from a model's set
# is dropped before the call to ``posterior_nn`` so downstream libraries
# (e.g. zuko) never receive unexpected keyword arguments.
_MODEL_KWARGS: dict[str, set[str]] = {
    # Classic sbi flows — accept the full PosteriorConfig surface
    "maf": {
        "hidden_features",
        "num_transforms",
        "dropout_probability",
        "num_blocks",
        "num_bins",
    },
    "nsf": {
        "hidden_features",
        "num_transforms",
        "dropout_probability",
        "num_blocks",
        "num_bins",
    },
    "mdn": {
        "hidden_features",
        "num_transforms",
        "dropout_probability",
        "num_blocks",
        "num_bins",
    },
    # Zuko-backed flows — num_blocks is a MAF/MLP concept not accepted by zuko
    "zuko_maf": {"hidden_features", "num_transforms"},
    "zuko_nsf": {"hidden_features", "num_bins"},
    "zuko_bpf": {"hidden_features", "num_transforms", "num_bins"},
    "zuko_ncsf": {"hidden_features", "num_transforms", "num_bins"},
    "zuko_nice": {"hidden_features", "num_transforms"},
    "zuko_gf": {"hidden_features", "num_transforms"},
    "zuko_unaf": {"hidden_features", "num_transforms"},
    "zuko_saf": {"hidden_features", "num_transforms"},
}

# Fallback: all kwargs. Unknown model names pass everything and let sbi raise.
_ALL_KWARGS: set[str] = {
    "hidden_features",
    "num_transforms",
    "dropout_probability",
    "num_blocks",
    "num_bins",
}


def _build_posterior_nn_kwargs(config: PosteriorConfig) -> dict:
    """
    Build the ``**kwargs`` dict for :func:`~sbi.neural_nets.posterior_nn`.

    Only kwargs that the target model family actually accepts are included.
    Unknown model names fall back to passing everything.

    :param config: The full posterior configuration.
    :returns: Dict of kwargs safe to pass to ``posterior_nn``.
    """
    accepted = _MODEL_KWARGS.get(config.model.lower(), _ALL_KWARGS)

    all_kwargs = {
        "hidden_features": config.hidden_features,
        "num_transforms": config.num_transforms,
        "dropout_probability": config.dropout_probability,
        "num_blocks": config.num_blocks,
        "num_bins": config.num_bins,
    }

    filtered = {k: v for k, v in all_kwargs.items() if k in accepted}
    dropped = set(all_kwargs) - set(filtered)
    if dropped:
        logger.debug(
            f"Model '{config.model}' does not accept {sorted(dropped)}; "
            "these kwargs were dropped from the posterior_nn call."
        )
    return filtered


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _load_checkpoint_dict(checkpoint_path: Path) -> dict:
    """Load a raw checkpoint dict from disk, mapping to CPU."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return cast(
        dict[Any, Any],
        torch.load(checkpoint_path, map_location="cpu", weights_only=False),
    )


def _extract_model_config(
    ckpt: dict, fallback: PosteriorConfig | None
) -> PosteriorConfig:
    """
    Pull the PosteriorConfig out of *ckpt*, falling back to *fallback*.

    :raises ValueError: If neither the checkpoint nor the fallback supplies a config.
    """
    config: PosteriorConfig | None = (
        ckpt.get("model_config") if isinstance(ckpt, dict) else None
    )
    if config is not None and fallback is not None:
        logger.warning(
            "Both a checkpoint model_config and a posterior_config were provided. "
            "The checkpoint config will be used — the supplied config is ignored."
        )
    if config is None:
        config = fallback
    if config is None:
        raise ValueError(
            "No model_config found in checkpoint and none was supplied. "
            "Pass posterior_config= explicitly for old checkpoints."
        )
    return config


def _strip_compiled_prefix(state_dict: dict) -> dict:
    """Remove the ``_orig_mod.`` prefix that torch.compile adds."""
    if any(k.startswith("_orig_mod.") for k in state_dict):
        logger.debug("Stripped _orig_mod. prefix from compiled model state dict")
        return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    return state_dict


def _infer_dims_from_state_dict(state_dict: dict, theta_dim: int) -> tuple[int, int]:
    embedding_keys = sorted(
        k for k in state_dict if "_embedding_net" in k and state_dict[k].ndim == 2
    )
    if embedding_keys:
        x_dim = state_dict[embedding_keys[0]].shape[1]
        logger.debug(
            f"Inferred x_dim={x_dim} from embedding net key '{embedding_keys[0]}'"
        )
        return theta_dim, x_dim

    hyper_keys = sorted(
        k for k in state_dict if "hyper.0.weight" in k and state_dict[k].ndim == 2
    )
    if hyper_keys:
        # Zuko hyper net input is x_dim + theta_dim (context is concatenated)
        combined_dim = state_dict[hyper_keys[0]].shape[1]
        x_dim = combined_dim - theta_dim
        logger.debug(
            f"Inferred x_dim={x_dim} from hyper net key '{hyper_keys[0]}' "
            f"(combined_dim={combined_dim} - theta_dim={theta_dim})"
        )
        return theta_dim, x_dim

    raise ValueError(
        "Could not infer x_dim from checkpoint state dict.\n"
        "Expected either '_embedding_net' keys or 'hyper.0.weight' keys.\n"
        f"Available keys (first 20): {list(state_dict.keys())[:20]}"
    )


# ---------------------------------------------------------------------------
# Trainer / callback factories
# ---------------------------------------------------------------------------
def _build_callbacks(config: TrainingConfig) -> list:
    """Construct the standard callback stack from *config*."""
    if config.save_path is None:
        raise ValueError("TrainingConfig.save_path must be set before training.")

    model_checkpoint = ModelCheckpoint(
        dirpath=config.save_path.parent,
        filename=f"{config.save_path.stem}_" + "{epoch}",
        monitor="val/ema_loss",
        save_top_k=3,
        every_n_epochs=config.autosave_every,
        save_last=True,
    )
    model_checkpoint.CHECKPOINT_NAME_LAST = str(config.save_path.stem)

    return [
        EarlyStopping(
            monitor="val/ema_loss", patience=config.stop_after_epochs, mode="min"
        ),
        model_checkpoint,
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # if config.prune_model is not None:
    #     model_pruning = ModelPruning("l1_unstructured", amount=config.prune_model)
    #     callbacks.append(model_pruning)
    # return callbacks


def _build_trainer(config: TrainingConfig) -> lightning.Trainer:
    """Construct a Lightning Trainer from *config*."""
    acc, strat = _select_accelerator_and_strategy()
    tb_logger = (
        TensorBoardLogger(save_dir=str(config.tensorboard_dir))
        if config.tensorboard_dir
        else True
    )
    return lightning.Trainer(
        max_epochs=config.max_epochs,
        callbacks=_build_callbacks(config),
        logger=tb_logger,
        precision="bf16-mixed" if config.use_amp else "32-true",
        gradient_clip_val=20.0,
        enable_progress_bar=config.show_progress,
        log_every_n_steps=50,
        strategy=strat,
        accelerator=acc,
        devices="auto",
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class InferenceHandler:
    """
    High-level interface for NPE training and posterior sampling.

    The three distinct lifecycles (fresh train / resume / infer-only) are
    each expressed as a clean call sequence — see module docstring.
    """

    def __init__(
        self,
        prior_path: Path,
        nuisance_pars: list[str] | None = None,
    ) -> None:
        """
        Initialise the handler and load the prior.

        :param prior_path: Path to a pickled :class:`~mach3sbitools.simulator.Prior`.
        :param nuisance_pars: fnmatch patterns for parameters to exclude.
        """
        self.device_handler = TorchDeviceHandler()
        self.prior = load_prior(prior_path).to(self.device_handler.device)
        self.parameter_names = self.prior.prior_data.parameter_names
        self.nuisance_pars = nuisance_pars

        if len(
            self.prior.prior_data[self.prior._nuisance_filter].parameter_names
        ) != len(self.prior.prior_data.parameter_names):
            raise ValueError(
                "Prior must have same nuisance params as inference handler!"
            )

        self.dataset: ParaketDataset | None = None
        self.inference: NPE | None = None
        self.posterior = None
        self._density_estimator: nn.Module | None = None
        self._tensor_dataset: TensorDataset | None = None

    # ── Data ──────────────────────────────────────────────────────────────────

    def set_dataset(self, data_folder: Path) -> None:
        """
        Point the handler at a folder of ``.feather`` simulation files.

        :param data_folder: Directory containing ``.feather`` files.
        """
        self.dataset = ParaketDataset(
            data_folder, self.parameter_names.tolist(), self.nuisance_pars
        )
        logger.info(
            f"Dataset set: [bold]{len(self.dataset)}[/] files in [cyan]{data_folder}[/]"
        )

    def load_training_data(self, verbose: bool = True) -> None:
        """..."""
        if self.dataset is None:
            raise ValueError("Call set_dataset() before load_training_data().")
        self._tensor_dataset = self.dataset.to_tensor_dataset(
            device="cpu", verbose=verbose
        )
        for t in self._tensor_dataset.tensors:
            t.share_memory_()

    # ── Model construction ────────────────────────────────────────────────────

    def create_posterior(self, config: PosteriorConfig) -> None:
        """
        Build the NPE inference object and density estimator network.

        Only the kwargs that the chosen model family actually accepts are
        forwarded to ``posterior_nn``; unsupported kwargs (e.g. ``num_blocks``
        for zuko-backed flows) are dropped with a DEBUG log line rather than
        raising a ``TypeError`` at runtime.

        :param config: Architecture and hyperparameter settings.
        """
        kwargs = _build_posterior_nn_kwargs(config)
        neural_net = posterior_nn(
            model=config.model,
            z_score_x="structured",
            z_score_theta="independent",
            **kwargs,
        )
        self.inference = NPE(
            prior=self.prior,
            density_estimator=neural_net,
            device=self.device_handler.device,
        )
        logger.info(
            f"NPE created | {config.model} | "
            f"hidden=[cyan]{config.hidden_features}[/] "
            f"transforms=[cyan]{config.num_transforms}[/] "
            f"blocks=[cyan]{config.num_blocks}[/] "
            f"bins=[cyan]{config.num_bins}[/]"
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def train_posterior(
        self,
        config: TrainingConfig,
        model_config: PosteriorConfig | None = None,
    ) -> None:
        """
        Train the density estimator from scratch using PyTorch Lightning.

        Requires :meth:`load_training_data` and :meth:`create_posterior` to
        have been called first.

        :param config: Training loop settings.
        :param model_config: Architecture config embedded in every checkpoint.
        :raises ValueError: If training data or the NPE object are missing.
        """
        if self._tensor_dataset is None:
            raise ValueError("Call load_training_data() before train_posterior().")
        if self.inference is None:
            raise ValueError("Call create_posterior() before train_posterior().")

        density_estimator = self._build_density_estimator_from_inference()
        self._fit(density_estimator, config, model_config, ckpt_path=None)

    def resume_training(
        self,
        checkpoint_path: Path,
        config: TrainingConfig,
    ) -> None:
        """
        Resume training from an existing checkpoint.

        Single entry-point for the ``--resume_checkpoint`` flow.  The model
        architecture is read directly from the checkpoint — :meth:`create_posterior`
        does **not** need to be called beforehand, and any ``--model`` /
        ``--hidden`` / etc. flags passed on the CLI are intentionally ignored
        to prevent silent architecture mismatches.

        ``_build_posterior_nn_kwargs`` filtering applies here exactly as it
        does for a fresh run, so zuko models resume without error regardless
        of which kwargs were stored in the checkpoint config.

        Requires :meth:`load_training_data` to have been called.

        :param checkpoint_path: Path to a ``.ckpt`` produced by a previous run.
        :param config: Training loop settings for the resumed run.
        :raises FileNotFoundError: If *checkpoint_path* does not exist.
        :raises ValueError: If training data has not been loaded.
        """
        if self._tensor_dataset is None:
            raise ValueError("Call load_training_data() before resume_training().")

        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Resuming training from [cyan]{checkpoint_path}[/]")

        ckpt = _load_checkpoint_dict(checkpoint_path)
        model_config = _extract_model_config(ckpt, fallback=None)

        self.create_posterior(model_config)
        density_estimator = self._build_density_estimator_from_inference()

        self._fit(
            density_estimator,
            config,
            model_config,
            ckpt_path=str(checkpoint_path),
        )

    def _build_density_estimator_from_inference(self) -> nn.Module:
        """
        Initialise the density estimator using the first batch of training data
        as a shape probe.
        """
        if self.inference is None:
            raise ValueError("inference is None — call create_posterior() first.")
        assert self._tensor_dataset is not None
        sample_theta = self._tensor_dataset.tensors[0][:10]
        sample_x = self._tensor_dataset.tensors[1][:10]
        return cast(nn.Module, self.inference._build_neural_net(sample_theta, sample_x))

    def _fit(
        self,
        density_estimator: nn.Module,
        config: TrainingConfig,
        model_config: PosteriorConfig | None,
        ckpt_path: str | None,
    ) -> None:
        """Internal: run the Lightning training loop."""
        assert self._tensor_dataset is not None

        lightning_module = SBILightningModule(density_estimator, config, model_config)

        # Compilation currently just seems really slow... (but adding it in for completeness!)
        if config.compile:
            logger.warning(
                "Requested model compilation. In testing this has been shown to be slower."
            )
            torch.compile(lightning_module)

        data_module = SBIDataModule(self._tensor_dataset, config)
        trainer = _build_trainer(config)

        # TODO: Uncomment when lightning allows for ddp batched training and LR with multiple optimizers
        # Currently LR decay is more effective than finding the perfect initial LR

        # Set up tuning to get good initial LR + batch size that uses the optimal amount of memory!
        # tuner = Tuner(trainer)
        # tuner.scale_batch_size(lightning_module, mode="power", datamodule=data_module)
        # tuner.lr_find(lightning_module, datamodule=data_module)

        trainer.fit(lightning_module, datamodule=data_module, ckpt_path=ckpt_path)

        self._density_estimator = lightning_module.model
        self._density_estimator.to(self.device_handler.device).eval()

        if config.save_path is None:
            raise ValueError(
                "TrainingConfig.save_path must be set to save the final model."
            )
        trainer.save_checkpoint(config.save_path)
        logger.info(f"Final checkpoint saved to [cyan]{config.save_path}[/]")

    # ── Posterior loading ─────────────────────────────────────────────────────

    def load_posterior(
        self,
        checkpoint_path: Path,
        posterior_config: PosteriorConfig | None = None,
    ) -> None:
        """
        Load a trained density estimator from a checkpoint for **inference only**.

        The ``PosteriorConfig`` is read from the checkpoint's ``"model_config"``
        key. ``_build_posterior_nn_kwargs`` filtering applies, so loading a
        zuko checkpoint works even if ``num_blocks`` is present in the stored
        config (it will simply be dropped).

        :param checkpoint_path: Path to a ``.pt`` / ``.ckpt`` checkpoint.
        :param posterior_config: Backwards-compat only; ignored when the
            checkpoint is self-contained.
        :raises FileNotFoundError: If *checkpoint_path* does not exist.
        :raises ValueError: If no model config can be determined.
        """
        checkpoint_path = Path(checkpoint_path)
        ckpt = _load_checkpoint_dict(checkpoint_path)

        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state_dict = _strip_compiled_prefix(ckpt["model_state"])
            model_config = _extract_model_config(ckpt, fallback=posterior_config)
            logger.info(
                f"Loading autosave checkpoint from epoch {ckpt.get('epoch', '?')}"
            )
        else:
            if posterior_config is None:
                raise ValueError(
                    "Plain state-dict checkpoint contains no model_config. "
                    "Pass posterior_config= explicitly."
                )
            state_dict = ckpt
            logger.info("Loading best-model state dict")
            model_config = posterior_config

        self.create_posterior(model_config)

        theta_dim = len(self.parameter_names)
        _, x_dim = _infer_dims_from_state_dict(state_dict, theta_dim)

        print(x_dim)

        device = self.device_handler.device
        density_estimator = self.inference._build_neural_net(  # type: ignore[union-attr]
            torch.zeros(2, theta_dim, device=device),
            torch.zeros(2, x_dim, device=device),
        )
        density_estimator.load_state_dict(state_dict)
        density_estimator.to(device).eval()
        self._density_estimator = density_estimator
        logger.info(f"Density estimator loaded from [cyan]{checkpoint_path}[/]")

    # ── Posterior sampling ────────────────────────────────────────────────────

    def build_posterior(self) -> None:
        """
        Wrap the trained density estimator in an ``sbi`` posterior object.

        :raises ValueError: If no density estimator or NPE object is present.
        """
        if self._density_estimator is None:
            raise ValueError("Train or load a density estimator first.")
        if self.inference is None:
            raise ValueError("Call create_posterior() before build_posterior().")
        pars = DirectPosteriorParameters(enable_transform=False)
        self.posterior = self.inference.build_posterior(
            self._density_estimator, posterior_parameters=pars
        )

    def sample_posterior(
        self,
        num_samples: int,
        x: list[float] | np.ndarray,
        **kwargs,
    ) -> torch.Tensor:
        """
        Draw samples from the posterior conditioned on *x*.

        :param num_samples: Number of posterior samples to draw.
        :param x: Observed data vector *x_o*.
        :returns: Tensor of shape ``(num_samples, n_params)``.
        """
        logger.info(f"Sampling [bold]{num_samples:,}[/] points from posterior")
        self.build_posterior()
        if self.posterior is None:
            raise ValueError("Train or load a density estimator first.")
        x_tensor = self.device_handler.to_tensor(x).to(self.prior.device_handler.device)
        return cast(
            torch.Tensor,
            self.posterior.sample((num_samples,), x=x_tensor, **kwargs),
        )

    def get_log_likelihood(
        self, theta: SimulatorData, x: list[float] | np.ndarray, **kwargs
    ) -> torch.Tensor:
        """
        Evaluate the log-likelihood of *theta* given observed data *x*.

        :param theta: Parameter array of shape ``(n_samples, n_params)``.
        :param x: Observed data vector *x_o*.
        :returns: Log-probability tensor of shape ``(n_samples,)``.
        """
        self.build_posterior()
        if self.posterior is None:
            raise ValueError("Train or load a density estimator first.")
        x_tensor = torch.tensor(
            np.array([x]), dtype=torch.float32, device=self.device_handler.device
        )
        theta_tensor = torch.tensor(
            np.array(theta), dtype=torch.float32, device=self.device_handler.device
        )
        return cast(
            torch.Tensor,
            cast(DirectPosterior, self.posterior).log_prob(
                theta=theta_tensor, x=x_tensor, **kwargs
            ),
        )
