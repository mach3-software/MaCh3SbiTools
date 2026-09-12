"""
HW: Code to perform inference
"""

import os
from pathlib import Path, PosixPath, WindowsPath
from typing import cast

import lightning
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from sbi.inference import NPE, DirectPosterior
from sbi.inference.posteriors.posterior_parameters import DirectPosteriorParameters
from sbi.neural_nets import posterior_nn
from sbi.samplers.rejection import rejection
from sbi.utils.user_input_checks import process_x
from tqdm.auto import tqdm

from mach3sbitools.data_loaders import SBIDataModule, TrainingDataset
from mach3sbitools.data_processors import (
    CompressorBase,
    compressor_factory,
    restore_compressor,
)
from mach3sbitools.simulator import CompressedPriorWrapper, load_prior
from mach3sbitools.types import SimulatorData
from mach3sbitools.utils import (
    PosteriorConfig,
    TrainingConfig,
    get_device,
    get_logger,
    to_tensor,
)

from .inference_utils import select_accelerator_and_strategy, select_model_kwargs
from .lightning_module import SBILightningModule
from .model_loader import ModelLoader

# Standard boiler plate
logger = get_logger()

# Rows drawn from the dataset when fitting compressors or probing shapes.
_PROBE_ROWS = 100_000
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
        self.device = get_device()
        self.prior = load_prior(prior_path, self.device)
        self.parameter_names = self.prior.prior_data.parameter_names

        self.dataset: TrainingDataset | None = None
        self.inference: NPE | None = None
        self.posterior = None
        self._density_estimator: nn.Module | None = None

        # Compression for X/Theta
        self._theta_compressor: CompressorBase | None = None
        self._x_compressor: CompressorBase | None = None

    def set_dataset(self, data_folder: Path) -> None:
        """
        Point the handler at a merged dataset folder.

        The folder must contain the ``theta.npy`` / ``x.npy`` pair written by
        :func:`~mach3sbitools.apps.merge_shards.merge_shards_module`.

        :param data_folder: Directory containing ``theta.npy`` and ``x.npy``.
        :raises FileNotFoundError: If either memmap file is missing.
        """
        x_data = data_folder / "x.npy"
        theta_data = data_folder / "theta.npy"

        if not x_data.is_file():
            raise FileNotFoundError(f"Cannot find x data file: {x_data}")

        if not theta_data.is_file():
            raise FileNotFoundError(f"Cannot find theta data file: {theta_data}")

        self.dataset = TrainingDataset(theta_data, x_data, self.prior)
        logger.info(
            f"Dataset set: [bold]{len(self.dataset):,}[/] rows "
            f"in [cyan]{data_folder}[/]"
        )

    def _probe_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Draw a representative sample of the dataset.

        Rows are sampled at random rather than taken from the head of the
        file: merged shards arrive grouped by source simulation, so a
        contiguous slice would bias both the compressor fit and the
        z-scoring statistics.

        :returns: Tuple of ``(theta, x)`` probe tensors.
        :raises ValueError: If no dataset has been set.
        """
        if self.dataset is None:
            raise ValueError("No dataset set — call set_dataset() first.")

        n_rows = len(self.dataset)
        n_probe = min(_PROBE_ROWS, n_rows)

        if n_probe == n_rows:
            indices = np.arange(n_rows)
        else:
            rng = np.random.default_rng(seed=42)
            indices = np.sort(rng.choice(n_rows, size=n_probe, replace=False))

        return self.dataset[indices]

    def fit_x_compressor(self, compressor: str, **kwargs) -> None:
        """
        Fit a compressor on the observable (``x``) dimension.

        The fitted compressor is applied to every training batch and to any
        ``x`` passed to :meth:`sample_posterior`, so the model is trained and
        conditioned in the same space.

        :param compressor: Name of a registered compressor, e.g. ``"pca"``.
        :param kwargs: Forwarded to the compressor's constructor.
        :raises ValueError: If no dataset has been set.
        """
        _, x = self._probe_batch()
        self._x_compressor = compressor_factory(compressor, **kwargs).fit(x)
        logger.info(f"Fitted x with {compressor}")

    def fit_theta_compressor(self, compressor: str, **kwargs) -> None:
        """
        Fit a compressor on the parameter (``theta``) dimension.

        :param compressor: Name of a registered compressor, e.g. ``"pca"``.
        :param kwargs: Forwarded to the compressor's constructor.
        :raises ValueError: If no dataset has been set.
        """
        theta, _ = self._probe_batch()
        self._theta_compressor = compressor_factory(compressor, **kwargs).fit(theta)
        logger.info(f"Fitted theta with {compressor}")

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
            device=self.device,
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
        if self.dataset is None:
            raise ValueError("Call set_dataset() before train_posterior().")
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
        Continue training from a checkpoint, reusing its architecture.

        Any compressors stored in the checkpoint are restored first, so
        training resumes in the same compressed space it left off in.

        :param checkpoint_path: Checkpoint to resume from.
        :param config: Training loop settings for the resumed run.
        """
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
        """
        Internal: run the Lightning training loop.

        :param density_estimator: Network to train.
        :param config: Training loop settings.
        :param model_config: Architecture config embedded in every checkpoint.
        :param ckpt_path: Checkpoint to resume the Lightning loop from.
        :raises ValueError: If no dataset has been set, or ``save_path`` is unset.
        """
        if self.dataset is None:
            raise ValueError("Call set_dataset() before training.")

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

        data_module = SBIDataModule(self.dataset, config)
        trainer = self._build_trainer(config)

        trainer.fit(lightning_module, datamodule=data_module, ckpt_path=ckpt_path)

        self._density_estimator = lightning_module.model
        self._density_estimator.to(self.device).eval()
        # Weights changed — force the next sample call to rebuild.
        self.posterior = None

        if config.save_path is None:
            raise ValueError(
                "TrainingConfig.save_path must be set to save the final model."
            )
        trainer.save_checkpoint(config.save_path)
        logger.info(f"Final checkpoint saved to [cyan]{config.save_path}[/]")

    # ================================================
    # Sampling
    # ================================================
    def build_posterior(self, rebuild: bool = False) -> None:
        """
        Wrap the trained density estimator in an ``sbi`` posterior object.

        The result is cached on :attr:`posterior`; repeated calls are no-ops
        unless *rebuild* is set or the estimator has since been retrained.

        :param rebuild: Force reconstruction even if a posterior is cached.
        :raises ValueError: If the density estimator or NPE object is missing.
        """
        if self.posterior is not None and not rebuild:
            return

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

        # enable_transform makes sbi build a bijection from the prior support.
        # The compressed prior's support is a custom constraint with no
        # registered bijection, and sampling goes through rejection rather than
        # MCMC, so the transform is both unbuildable and unnecessary there.
        pars = DirectPosteriorParameters(
            enable_transform=self._theta_compressor is None
        )
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
        """
        Draw posterior samples conditioned on the observation *x*.

        Sampling is rejection-based against the true prior support, so every
        returned row lies within bounds. If a theta compressor is active the
        samples are decompressed before being returned, i.e. the caller
        always sees the original parameter space.

        :param num_samples: Number of samples to draw.
        :param x: Observation(s) to condition on.
        :param kwargs: Rejection-sampler overrides — ``show_progress_bars``,
            ``max_sampling_batch_size``, ``max_sampling_time``.
        :returns: Samples of shape ``(num_samples, theta_dim)`` for a single
            observation, or ``(num_samples, n_observations, theta_dim)`` when
            *x* holds several.
        :raises ValueError: If no density estimator has been trained or loaded.
        """
        logger.info(f"Sampling [bold]{num_samples:,}[/] points from posterior")
        self.build_posterior()
        if self.posterior is None:
            raise ValueError("Train or load a density estimator first.")

        x_tensor = to_tensor(x, self.device)

        # The rejection sampler always emits a condition axis. Drop it again
        # for a single observation so callers get (num_samples, theta_dim).
        single_observation = x_tensor.ndim == 1

        if self._x_compressor is not None:
            x_tensor = self._x_compressor.transform(x_tensor).to(self.device)

        # ── Define the vectorized boundary force ──────────────────────────────
        def strict_prior_mask(theta_proposed: torch.Tensor) -> torch.Tensor:
            """
            Accept only proposals inside the true prior support.

            :param theta_proposed: Proposals in the model's (possibly
                compressed) parameter space.
            :returns: Boolean acceptance mask.
            """
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

        # Posterior samples arrive in compressed space; decompress before returning.
        samples_compressed = rejection.accept_reject_sample(
            proposal=self.posterior.posterior_estimator.sample,
            accept_reject_fn=strict_prior_mask,
            num_samples=num_samples,
            show_progress_bars=kwargs.get("show_progress_bars", True),
            max_sampling_batch_size=kwargs.get("max_sampling_batch_size", 10_000),
            proposal_sampling_kwargs={"condition": process_x(x_tensor)},
            alternative_method="build_posterior(..., sample_with='mcmc')",
            max_sampling_time=kwargs.get("max_sampling_time", None),
            return_partial_on_timeout=True,
        )[0]

        if self._theta_compressor is not None:
            samples = self._theta_compressor.inverse_transform(samples_compressed)
        else:
            samples = samples_compressed

        return samples.squeeze(1) if single_observation else samples

    def get_log_likelihood(
        self, theta: SimulatorData, x: list[float] | np.ndarray, **kwargs
    ) -> torch.Tensor:
        """
        Evaluate the log-likelihood of *theta* given observed data *x*.

        :param theta: Parameter array of shape ``(n_samples, n_params)``.
        :param x: Observed data vector *x_o*.
        :param kwargs: Forwarded to the posterior's ``log_prob``.
        :returns: Log-probability tensor of shape ``(n_samples,)``.
        :raises ValueError: If no density estimator has been trained or loaded.
        """
        self.build_posterior()
        if self.posterior is None:
            raise ValueError("Train or load a density estimator first.")
        x_tensor = torch.tensor(np.array([x]), dtype=torch.float32, device=self.device)
        theta_tensor = torch.tensor(
            np.array(theta), dtype=torch.float32, device=self.device
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

    def _load_posterior(self, loader: ModelLoader) -> None:
        """
        Rebuild the network described by *loader* and load its weights.

        :param loader: Reader over an already-opened checkpoint.
        """
        self.create_posterior(loader.model_config)

        device = self.device
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
        # New weights — invalidate any cached posterior.
        self.posterior = None

    # ================================================
    # Builders
    # ================================================
    def _build_callbacks(self, config: TrainingConfig) -> list:
        """
        Construct the standard callback stack from *config*.

        :param config: Training loop settings.
        :returns: Early stopping, checkpointing and LR monitoring callbacks.
        :raises ValueError: If ``config.save_path`` is unset.
        """
        if config.save_path is None:
            raise ValueError("TrainingConfig.save_path must be set before training.")

        # Everything decides on the raw validation loss. `val/ema_loss` lags
        # the true optimum by roughly (1 - alpha) / alpha epochs, which would
        # both delay stopping and make the kept checkpoint a later, worse one.
        # EarlyStopping's own patience already provides noise tolerance, and
        # it compares against the best value seen rather than a lagged average.
        model_checkpoint = ModelCheckpoint(
            dirpath=config.save_path.parent,
            filename=f"{config.save_path.stem}_" + "{epoch}",
            monitor="val/loss",
            save_top_k=3,
            every_n_epochs=config.autosave_every,
            save_last=True,
        )
        model_checkpoint.CHECKPOINT_NAME_LAST = str(config.save_path.stem)  # type: ignore

        return [
            EarlyStopping(
                monitor="val/loss",
                patience=config.stop_after_epochs,
                min_delta=config.min_delta,
                mode="min",
            ),
            model_checkpoint,
            LearningRateMonitor(logging_interval="epoch"),
        ]

    def _build_density_estimator_from_inference(self) -> nn.Module:
        """
        Build the network, sizing and z-scoring it in the training space.

        The probe batch is pushed through the fitted compressors first, so the
        network's input dimensions and z-score statistics match the batches
        :class:`~mach3sbitools.inference.SBILightningModule` will feed it.

        :returns: An untrained density estimator.
        :raises ValueError: If :meth:`create_posterior` has not been called.
        """
        if self.inference is None:
            raise ValueError("inference is None — call create_posterior() first.")

        # A large representative batch keeps the z-score statistics accurate.
        sample_theta, sample_x = self._probe_batch()

        if self._theta_compressor is not None:
            sample_theta = self._theta_compressor.transform(sample_theta)
        if self._x_compressor is not None:
            sample_x = self._x_compressor.transform(sample_x)

        logger.info(
            "Building density estimator — theta dim: %d | x dim: %d",
            sample_theta.shape[-1],
            sample_x.shape[-1],
        )
        return cast(nn.Module, self.inference._build_neural_net(sample_theta, sample_x))

    def _build_trainer(self, config: TrainingConfig) -> lightning.Trainer:
        """
        Construct a Lightning Trainer from *config*.

        :param config: Training loop settings.
        :returns: A Trainer wired up with the selected accelerator and strategy.
        """
        accelerator, strategy = select_accelerator_and_strategy(
            use_model_parallel=False
        )
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
            limit_train_batches=config.limit_train_batches,
            limit_val_batches=config.limit_val_batches,
            strategy=strategy,
            accelerator=accelerator,
            devices="auto",
            num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
            num_sanity_val_steps=0,
        )

    def sample_posterior_chunked(
        self,
        num_samples_per_x: int,
        x: list[float] | np.ndarray,
        chunk_size: int = 500,
        show_chunk_progress: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Draw `num_samples_per_x` posterior sample(s) for each row of `x`,
        processing `chunk_size` observations at a time so the rejection
        sampler's internal (max_sampling_batch_size x n_conditions) broadcast
        stays bounded regardless of how large `x` or `num_samples_per_x` are.

        :param num_samples_per_x: Samples drawn for each row of *x*.
        :param x: Observations of shape ``(n_observations, x_dim)``.
        :param chunk_size: Observations processed per rejection-sampling pass.
        :param show_chunk_progress: Show a progress bar over the chunks.
        :param kwargs: Forwarded to :meth:`sample_posterior`.
        :returns: Tensor of shape ``(n_observations, num_samples_per_x, theta_dim)``,
            or ``(n_observations, theta_dim)`` if ``num_samples_per_x == 1``.
        """
        x_arr = np.asarray(x)
        n_obs = x_arr.shape[0]
        kwargs.setdefault("show_progress_bars", False)

        chunk_iter = range(0, n_obs, chunk_size)
        if show_chunk_progress:
            chunk_iter = tqdm(
                chunk_iter,
                desc=f"Sampling posterior in chunks of {chunk_size}",
                total=(n_obs + chunk_size - 1) // chunk_size,
            )

        chunks = []
        for start in chunk_iter:
            end = min(start + chunk_size, n_obs)
            x_chunk = x_arr[start:end]
            samples_chunk = self.sample_posterior(
                num_samples_per_x, x=x_chunk, **kwargs
            ).cpu()
            # shape (num_samples_per_x, chunk_len, theta_dim)
            chunks.append(samples_chunk)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Reassemble the full observation axis (was split across chunks on dim=1)
        samples = torch.cat(chunks, dim=1)  # (num_samples_per_x, n_obs, theta_dim)

        # (n_obs, num_samples_per_x, theta_dim) is the more natural layout —
        # matches thetas/xs indexing by observation.
        samples = samples.permute(1, 0, 2)

        if num_samples_per_x == 1:
            samples = samples.squeeze(1)  # (n_obs, theta_dim), for LC2ST-style calls

        return samples
