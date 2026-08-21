"""Train application module."""

import os
import warnings
from pathlib import Path

import torch

from mach3sbitools.inference import InferenceHandler
from mach3sbitools.utils import PosteriorConfig, TrainingConfig, get_logger

# Suppress the LeafSpec deprecation warnings stemming from lightning/torch interaction
# 1. Suppress the LeafSpec deprecation warnings stemming from torch/lightning pytree
warnings.filterwarnings("ignore", category=UserWarning, module=".*_pytree.*")
warnings.filterwarnings("ignore", message=".*LeafSpec.*")
warnings.filterwarnings("ignore", message=".*It is recommended to use.*")


def train_module(
    save_file: Path,
    prior_path: Path,
    dataset: Path,
    model: str,
    hidden: int,
    dropout: float,
    num_blocks: int,
    transforms: int,
    num_bins: int,
    batch_size: int,
    max_epochs: int,
    ema_alpha: float,
    learning_rate: float,
    stop_after_epochs: int,
    validation_fraction: float,
    num_workers: int,
    autosave_every: int,
    resume_checkpoint: Path | None,
    use_amp: bool,
    print_interval: int,
    tensorboard_dir: Path | None,
    scheduler_patience: int,
    show_progress: bool,
    compile_model: bool,
    prune_model: float | None,
    compress_x: bool,
    compress_theta: bool,
    compress_x_components: int,
    compress_theta_components: int,
) -> None:
    """Train a Neural Posterior Estimation (NPE) density estimator.

    Loads simulations from ``--dataset``, builds an NPE model with the
    specified architecture, and trains it with a custom loop featuring
    linear warm-up, ReduceLROnPlateau scheduling, EMA-based early stopping,
    and periodic checkpointing.

    When ``--resume_checkpoint`` is supplied the architecture is read directly
    from that checkpoint — no ``--model`` / ``--hidden`` / etc. flags are
    needed and any that are passed are silently ignored.

    Example::

        mach3sbi train \\
            -r prior.pkl -d sims/ -s models/run.pt \\
            --model maf --hidden 128 --transforms 8 \\
            --max_epochs 50000 --stop_after_epochs 200

    Resume::

        mach3sbi train \\
            -r prior.pkl -d sims/ -s models/run.pt \\
            --resume_checkpoint models/last.ckpt \\
            --max_epochs 50000 --stop_after_epochs 200
    """
    logger = get_logger()

    save_file = Path(save_file)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    training_config = TrainingConfig(
        save_path=save_file,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        stop_after_epochs=stop_after_epochs,
        validation_fraction=validation_fraction,
        num_workers=num_workers,
        autosave_every=autosave_every,
        resume_checkpoint=Path(resume_checkpoint) if resume_checkpoint else None,
        use_amp=use_amp,
        print_interval=print_interval,
        show_progress=show_progress,
        tensorboard_dir=Path(tensorboard_dir) if tensorboard_dir else None,
        scheduler_patience=scheduler_patience,
        compile=compile_model,
        ema_alpha=ema_alpha,
        prune_model=prune_model,
    )

    # Dataset loading is always required — shared CPU tensor, single load.
    handler = InferenceHandler(Path(prior_path))
    handler.set_dataset(Path(dataset))
    # All ranks must wait for rank 0 to finish loading before proceeding.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if resume_checkpoint:
        # ── Resume path ────────────────────────────────────────────────────
        # Architecture is read from the checkpoint; user-supplied model flags
        # are intentionally not used here to avoid silent mismatches.
        logger.info(f"Resuming from checkpoint: [cyan]{resume_checkpoint}[/]")
        handler.resume_training(Path(resume_checkpoint), training_config)
    else:
        # ── Fresh training path ────────────────────────────────────────────
        if compress_theta:
            handler.fit_theta_compressor("pca", n_components=compress_theta_components)

        if compress_x:
            handler.fit_x_compressor("pca", n_components=compress_x_components)

        posterior_config = PosteriorConfig(
            model=model,
            hidden_features=hidden,
            num_transforms=transforms,
            dropout_probability=dropout,
            num_blocks=num_blocks,
            num_bins=num_bins,
        )
        handler.create_posterior(posterior_config)
        handler.train_posterior(training_config, model_config=posterior_config)
