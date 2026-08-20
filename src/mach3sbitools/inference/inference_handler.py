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
from sbi.samplers.rejection import rejection
from sbi.utils.user_input_checks import process_x
from torch.utils.data import Dataset

from mach3sbitools.data_loaders import CompressedDataset, SBIDataModule, TrainingDataset
from mach3sbitools.data_processors import (
    CompressorBase,
    compressor_factory,
    restore_compressor,
)
from mach3sbitools.simulator import CompressedPriorWrapper, load_prior
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
    ) -> None:
        """
        Initialise the handler and load the prior.

        :param prior_path: Path to a pickled :class:`~mach3sbitools.simulator.Prior`.
        """
        self.device_handler = TorchDeviceHandler()
        self.prior = load_prior(prior_path)  # .to(self.device_handler.device)
        self.parameter_names = self.prior.prior_data.parameter_names

        # if len(
        #     self.prior.prior_data[self.prior._nuisance_filter].parameter_names
        # ) != len(self.prior.prior_data.parameter_names):
        #     raise ValueError(
        #         "Prior must have same nuisance params as inference handler!"
        #     )

        self.dataset: TrainingDataset | None = None
        self.inference: NPE | None = None
        self.posterior = None
        self._density_estimator: nn.Module | None = None
        # The dataset actually handed to SBIDataModule for training. Starts
        # out as the raw lazily-loading TrainingDataset (set in
        # load_training_data); becomes a CompressedDataset wrapper once
        # _apply_compression() runs. Never a fully-materialized tensor --
        # rows are read on demand by DataLoader workers.
        self._training_dataset: Dataset | None = None

        # Compression for X/Theta
        self._theta_compressor: CompressorBase | None = None
        self._x_compressor: CompressorBase | None = None

    def set_dataset(self, data_folder: Path) -> None:
        """
        Point the handler at a folder of ``.feather`` simulation files.

        :param data_folder: Directory containing ``.feather`` files.
        """
        self.dataset = TrainingDataset(data_folder, self.prior)
        logger.info(
            f"Dataset set: [bold]{len(self.dataset)}[/] files in [cyan]{data_folder}[/]"
        )

    def load_training_data(self, verbose: bool = True) -> None:
        """
        Prepare the training dataset for use.

        This used to materialize every row into a single in-RAM
        ``TensorDataset`` -- fine for small datasets, but it doesn't scale
        to O(1TB) simulation sets against a bounded RAM budget. It now just
        points training at the lazily-loading, memory-mapped dataset built
        by :meth:`set_dataset`: individual rows are read on demand by
        ``DataLoader`` workers (mmap + OS page cache) instead of being
        pre-loaded here. Kept as an explicit step, separate from
        :meth:`set_dataset`, for backward compatibility with existing call
        sites (e.g. ``train.py`` calling it once per-rank behind a barrier).

        :param verbose: Log dataset size on completion.
        """
        if self.dataset is None:
            raise ValueError("Call set_dataset() before load_training_data().")

        self._training_dataset = self.dataset
        if verbose:
            n_files = len(self.dataset.files)
            logger.info(
                f"Training data ready: [bold]{len(self.dataset):,}[/] rows "
                f"across [cyan]{n_files}[/] files "
                "(lazily loaded via mmap, not materialized in RAM)"
            )

    def fit_x_compressor(self, compressor: str, **kwargs):
        """
        Compress X dim.

        Fits on a bounded random subsample drawn directly from the lazy
        dataset (not the full O(1TB) set) -- PCA components stabilize long
        before the whole dataset is needed. Pass ``subsample=N`` to control
        how many rows are drawn; defaults to the compressor's own default
        (2,000,000 for PCA).
        """
        if self.dataset is None:
            raise ValueError("call set_dataset before fitting compressor")

        n_rows = kwargs.get("subsample", 2_000_000)
        _, x = self.dataset.random_subsample(n_rows)
        self._x_compressor = compressor_factory(compressor, **kwargs).fit(x)
        logger.info(f"Fitted x with {compressor} on {x.shape[0]:,} subsampled rows")

    def fit_theta_compressor(self, compressor: str, **kwargs):
        """
        Compress theta dim.

        See :meth:`fit_x_compressor` -- fits on a bounded random subsample
        rather than the full dataset.
        """
        if self.dataset is None:
            raise ValueError("call set_dataset before fitting compressor")

        n_rows = kwargs.get("subsample", 2_000_000)
        theta, _ = self.dataset.random_subsample(n_rows)
        self._theta_compressor = compressor_factory(compressor, **kwargs).fit(theta)
        logger.info(
            f"Fitted theta with {compressor} on {theta.shape[0]:,} subsampled rows"
        )

    def _apply_compression(self) -> None:
        """
        Wire fitted compressors in as an on-the-fly transform.

        Unlike the old in-place tensor transform, this never materializes a
        compressed copy of the dataset: each row is compressed at read
        time by :class:`~mach3sbitools.data_loaders.CompressedDataset`,
        which wraps whatever ``self._training_dataset`` currently is
        (typically the lazy, memory-mapped ``TrainingDataset``).
        """
        if self._training_dataset is None:
            raise ValueError("call load_training_data before applying compression")

        if self._theta_compressor is None and self._x_compressor is None:
            logger.info("No compressors fitted; training on raw theta/x.")
            return

        self._training_dataset = CompressedDataset(
            self._training_dataset,
            theta_compressor=self._theta_compressor,
            x_compressor=self._x_compressor,
        )
        logger.info(
            "Compression wired in as an on-the-fly transform | "
            f"theta_compressor={'yes' if self._theta_compressor else 'no'} "
            f"x_compressor={'yes' if self._x_compressor else 'no'}"
        )

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
            z_score_x="independent",
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
        if self._training_dataset is None:
            raise ValueError("Call load_training_data() before train_posterior().")
        if self.inference is None:
            raise ValueError("Call create_posterior() before train_posterior().")

        self._apply_compression()
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
        assert self._training_dataset is not None

        lightning_module = SBILightningModule(
            density_estimator,
            config,
            model_config,
            self._x_compressor,
            self._theta_compressor,
        )

        # Compilation currently just seems really slow... (but adding it in for completeness!)
        if config.compile:
            logger.warning(
                "Requested model compilation. In testing this has been shown to be slower."
            )
            torch.compile(lightning_module)

        data_module = SBIDataModule(self._training_dataset, config)
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
    # inference/inference_handler.py  — only the two methods below change

    def build_posterior(self) -> None:
        if self._density_estimator is None:
            raise ValueError("Train or load a density estimator first.")
        if self.inference is None:
            raise ValueError("Call create_posterior() before build_posterior().")

        # If theta was compressed during training, sbi must see the compressed
        # prior so that its support checks operate in the right space.
        if self._theta_compressor is not None:
            prior_for_sbi = CompressedPriorWrapper(self.prior, self._theta_compressor)
            # Temporarily swap the prior on the NPE object so build_posterior
            # picks up the wrapped version.
            original_prior = self.inference._prior
            self.inference._prior = prior_for_sbi
        else:
            original_prior = None

        pars = DirectPosteriorParameters(enable_transform=True)
        self.posterior = self.inference.build_posterior(
            self._density_estimator, posterior_parameters=pars
        )

        # Restore the real prior so the NPE object stays consistent for
        # any subsequent training or reloading.
        if original_prior is not None:
            self.inference._prior = original_prior

    def sample_posterior(
        self,
        num_samples: int,
        x: list[float] | np.ndarray,
        **kwargs,
    ) -> torch.Tensor:
        logger.info(f"Sampling [bold]{num_samples:,}[/] points from posterior")
        self.build_posterior()
        if self.posterior is None:
            raise ValueError("Train or load a density estimator first.")

        x_tensor = self.device_handler.to_tensor(x).to(self.device_handler.device)
        if self._x_compressor is not None:
            x_tensor = self._x_compressor.transform(x_tensor).to(
                self.device_handler.device
            )

        # ── Define the vectorized boundary force ──────────────────────────────
        def strict_prior_mask(theta_proposed: torch.Tensor) -> torch.Tensor:
            """Prior mask check"""
            # If the flow is working in compressed space, decompress it to check actual bounds
            if self._theta_compressor is not None:
                theta_checking = self._theta_compressor.inverse_transform(
                    theta_proposed
                )
            else:
                theta_checking = theta_proposed

            # prior.check_bounds returns a tensor of 1s and 0s (on its own device)
            # Convert to boolean and ensure it aligns with the sample's device
            return (
                self.prior.check_bounds(theta_checking).bool().to(theta_proposed.device)
            )

        # Inject rejection arguments into kwargs if not already overridden
        kwargs.setdefault("sample_with", "rejection")
        kwargs.setdefault("accept_reject_fn", strict_prior_mask)

        # Posterior samples arrive in compressed space; decompress before returning.
        # samples_compressed = cast(
        #     torch.Tensor,
        #     self.posterior.sample((num_samples,), x=x_tensor, **kwargs),
        # )

        samples_compressed = rejection.accept_reject_sample(
            proposal=self.posterior.posterior_estimator.sample,
            accept_reject_fn=lambda theta: strict_prior_mask(theta),
            num_samples=num_samples,
            show_progress_bars=kwargs.get("show_progress_bars", True),
            max_sampling_batch_size=kwargs.get("max_sampling_batch_size", 10_000),
            proposal_sampling_kwargs={"condition": process_x(x_tensor)},
            alternative_method="build_posterior(..., sample_with='mcmc')",
            max_sampling_time=kwargs.get("max_sampling_time", None),
            return_partial_on_timeout=True,
        )[0][:, 0]

        if self._theta_compressor is not None:
            return self._theta_compressor.inverse_transform(samples_compressed)
        return samples_compressed

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

        if self._x_compressor:
            x_tensor = self._x_compressor.transform(x_tensor)
        if self._theta_compressor:
            theta_tensor = self._theta_compressor.transform(theta_tensor)

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

        device = self.device_handler.device
        density_estimator = self.inference._build_neural_net(  # type: ignore[union-attr]
            torch.zeros(2, loader.theta_dim, device=device),
            torch.zeros(2, loader.x_dim, device=device),
        )
        density_estimator.load_state_dict(loader.state_dict)

        if x_comp := loader.x_compressor:
            self._x_compressor = restore_compressor(x_comp)

        if theta_comp := loader.theta_compressor:
            self._theta_compressor = restore_compressor(theta_comp)

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
        if self.inference is None:
            raise ValueError("inference is None — call create_posterior() first.")
        assert self.dataset is not None

        # Use a large representative batch for accurate z-score statistics
        # (10 samples, the previous default, gives wildly inaccurate mean/std).
        # Drawn as a bounded subsample from the lazy dataset rather than
        # sliced from a fully-materialized tensor.
        n_probe = min(100_000, len(self.dataset))
        sample_theta, sample_x = self.dataset.random_subsample(n_probe)

        # _apply_compression() only affects self._training_dataset (the
        # on-the-fly wrapper); replicate the same transform here so the
        # z-score probe sees data in the same space the network will train on.
        if self._theta_compressor is not None:
            sample_theta = self._theta_compressor.transform(sample_theta)
        if self._x_compressor is not None:
            sample_x = self._x_compressor.transform(sample_x)

        return cast(nn.Module, self.inference._build_neural_net(sample_theta, sample_x))

    def _build_trainer(self, config: TrainingConfig) -> lightning.Trainer:
        """Construct a Lightning Trainer from *config*."""
        acc, strat = select_accelerator_and_strategy(use_model_parallel=True)
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
            num_nodes=int(os.environ.get("§URM_NNODES", 1)),
        )
