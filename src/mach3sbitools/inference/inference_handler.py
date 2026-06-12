"""
HW: Code to perform inference
"""

# builtin
import os
from pathlib import Path, PosixPath, WindowsPath
from typing import cast

import lightning

# Non-builtin but standard
import numpy as np

# Torch
import torch
import torch.nn as nn

# Lighting
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger

# SBI
from sbi.inference import NPE, DirectPosterior
from sbi.inference.posteriors.posterior_parameters import DirectPosteriorParameters
from sbi.neural_nets import posterior_nn
from torch.utils.data import TensorDataset

from mach3sbitools.data_loaders import ParaketDataset, SBIDataModule
from mach3sbitools.simulator import load_prior
from mach3sbitools.types import SimulatorData

# SBI Tools
from mach3sbitools.utils import (
    PosteriorConfig,
    TorchDeviceHandler,
    TrainingConfig,
    get_logger,
)

from .inference_utils import select_accelerator_and_strategy, select_model_kwargs
from .lightning_module import SBILightningModule
from .model_loader import ModelLoader

# Standard boiler plate
logger = get_logger()
torch.set_float32_matmul_precision("medium")

torch.serialization.add_safe_globals(
    [
        TrainingConfig,
        PosteriorConfig,
        PosixPath,
        WindowsPath,
        Path,
    ]
)


class InferenceHandler:
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

    def create_posterior(self, config: PosteriorConfig) -> None:
        """
        Build the NPE inference object and density estimator network.

        Only the kwargs that the chosen model family actually accepts are
        forwarded to ``posterior_nn``; unsupported kwargs (e.g. ``num_blocks``
        for zuko-backed flows) are dropped with a DEBUG log line rather than
        raising a ``TypeError`` at runtime.

        :param config: Architecture and hyperparameter settings.
        """
        kwargs = select_model_kwargs(config)
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
        model_loader = ModelLoader(checkpoint_path)

        self._load_posterior(model_loader)

        assert self._density_estimator is not None

        self._fit(
            self._density_estimator,
            config,
            model_loader.model_config,
            str(checkpoint_path),
        )

    # ================================================
    # Internal Methods
    # ================================================
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
        trainer = self._build_trainer(config)

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

    # ================================================
    # Sampling
    # ================================================
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

    # ================================================
    # Loading
    # ================================================

    def load_posterior(self, checkpoint_path: Path):
        """
        Load a trained density estimator from a checkpoint for **inference only**.

        The ``PosteriorConfig`` is read from the checkpoint's ``"model_config"``
        key. ``_build_posterior_nn_kwargs`` filtering applies, so loading a
        zuko checkpoint works even if ``num_blocks`` is present in the stored
        config (it will simply be dropped).

        :param checkpoint_path: Path to a ``.pt`` / ``.ckpt`` checkpoint.
        :raises FileNotFoundError: If *checkpoint_path* does not exist.
        :raises ValueError: If no model config can be determined.
        """

        loader = ModelLoader(checkpoint_path)

        self._load_posterior(loader)
        logger.info(f"Density estimator loaded from [cyan]{checkpoint_path}[/]")

    def _load_posterior(self, loader: ModelLoader):
        self.create_posterior(loader.model_config)
        if loader.theta_dim != len(self.parameter_names):
            raise ValueError(
                f"Total number of parameters ({len(self.parameter_names)})!=theta_dim in checkpoint ({loader.theta_dim})"
            )

        device = self.device_handler.device
        density_estimator = self.inference._build_neural_net(  # type: ignore[union-attr]
            torch.zeros(2, loader.theta_dim, device=device),
            torch.zeros(2, loader.x_dim, device=device),
        )
        density_estimator.load_state_dict(loader.state_dict)

        density_estimator.to(device).eval()
        self._density_estimator = density_estimator

    # ================================================
    # Builders
    # ================================================
    def _build_callbacks(self, config: TrainingConfig) -> list:
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
        model_checkpoint.CHECKPOINT_NAME_LAST = str(config.save_path.stem)  # type: ignore

        return [
            EarlyStopping(
                monitor="val/ema_loss", patience=config.stop_after_epochs, mode="min"
            ),
            model_checkpoint,
            LearningRateMonitor(logging_interval="epoch"),
        ]

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

    def _build_trainer(self, config: TrainingConfig) -> lightning.Trainer:
        """Construct a Lightning Trainer from *config*."""
        acc, strat = select_accelerator_and_strategy()
        tb_logger = (
            TensorBoardLogger(save_dir=str(config.tensorboard_dir))
            if config.tensorboard_dir
            else True
        )
        return lightning.Trainer(
            max_epochs=config.max_epochs,
            callbacks=self._build_callbacks(config),
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
