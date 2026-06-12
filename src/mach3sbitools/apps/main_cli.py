from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
from click_option_group import optgroup

from mach3sbitools.utils import (
    MaCh3Logger,
)

from .diagnostics import diagnostics_module
from .importance_sample import importance_sample_module
from .inference import inference as inference_module
from .save_data import save_data_module
from .save_prior import save_prior_module
from .simulate import simulate_module
from .train import train_module


# ── Shared option helpers ──────────────────────────────────────────────────────
def apply_options(
    options: list[Any],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Generic decorator factory for any list of click options."""

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        for option in reversed(options):
            f = option(f)
        return f

    return decorator


_LOGGER_OPTIONS = [
    click.option(
        "--log-level",
        help="Console logging level. One of DEBUG, INFO, WARNING, ERROR.",
        default="INFO",
    ),
    click.option("--log_file", help="Optional path to write a plain-text log file."),
]

_SIMULATOR_OPTIONS = [
    click.option(
        "--output_file", "-o", help="Name of file to output to", required=True
    ),
    click.option(
        "--simulator_module",
        "-m",
        help="Dotted Python module path containing the simulator class (e.g. 'mypackage.simulator').",
        required=True,
    ),
    click.option(
        "--simulator_class",
        "-s",
        help="Name of the simulator class within the module. Must implement SimulatorProtocol.",
        required=True,
    ),
    click.option(
        "--config",
        "-c",
        type=Path,
        help="Path to the simulator configuration file (e.g. a MaCh3 fitter YAML).",
        required=True,
    ),
    click.option(
        "--nuisance_pars",
        "-p",
        multiple=True,
        help="Parameter name patterns (fnmatch-style, e.g. 'syst_*') to exclude from the prior and training.",
    ),
    click.option(
        "--cyclical_pars",
        "--cy",
        type=Path,
        multiple=True,
        help="Parameter name patterns (fnmatch-style) that should use a cyclical sinusoidal prior over [-2π, 2π].",
    ),
    click.option(
        "--flipped_pars",
        "--fl",
        type=Path,
        multiple=True,
        help="Parameter name patterns (fnmatch-style) that should flip around 0",
    ),
]


# ── CLI root ───────────────────────────────────────────────────────────────────


@click.group()
@apply_options(_LOGGER_OPTIONS)
def cli(log_file: Path | None, log_level: str) -> None:
    """mach3sbi — simulation-based inference tools for MaCh3.

    Run ``mach3sbi COMMAND --help`` for detailed usage of each subcommand.
    """
    MaCh3Logger(
        name="mach3sbi",
        level=log_level,
        log_file=Path(log_file) if log_file else None,
    )


# ── create_prior ──────────────────────────────────────────────────────────────
@cli.command(
    "create_prior", short_help="Generate and save a prior from a simulator instance."
)
@apply_options(_SIMULATOR_OPTIONS)
def save_prior(
    simulator_module: str,
    simulator_class: str,
    config: Path,
    output_file: Path,
    nuisance_pars: list[str],
    cyclical_pars: list[str],
    flipped_pars: list[str],
) -> None:
    """Generate a Prior from a simulator and save it to disk.

    Instantiates the simulator, reads its parameter names, bounds, nominals,
    and covariance, then constructs and pickles a Prior object ready for use
    in training and inference.

    Example::

        mach3sbi create_prior \\
            -m mypackage.simulator -s MySimulator \\
            -c config.yaml -o prior.pkl
    """
    save_prior_module(
        simulator_module,
        simulator_class,
        config,
        output_file,
        nuisance_pars,
        cyclical_pars,
        flipped_pars,
    )


# ── simulate ──────────────────────────────────────────────────────────────────
@cli.command("simulate", short_help="Run simulations and save to feather files.")
@apply_options(_SIMULATOR_OPTIONS)
@click.option(
    "--n_simulations",
    "-n",
    required=True,
    type=int,
    help="Number of simulation samples to generate.",
)
@click.option(
    "--prior_file",
    "-r",
    type=Path,
    help="Optional path to also save the prior used for this run.",
)
def simulate(
    simulator_module: str,
    simulator_class: str,
    config: Path,
    n_simulations: int,
    output_file: Path,
    nuisance_pars: list[str],
    cyclical_pars: list[str],
    flipped_pars: list[str],
    prior_file: Path | None,
) -> None:
    """Draw samples from the prior and run the simulator for each.

    Samples ``n_simulations`` parameter vectors θ from the prior, passes each
    through the simulator, applies Poisson smearing to the output, and saves
    the (θ, x) pairs as a feather file. Failed simulations are skipped with a
    warning.

    Example::

        mach3sbi simulate \\
            -m mypackage.simulator -s MySimulator \\
            -c config.yaml -n 100000 -o sims.feather
    """
    simulate_module(
        simulator_module,
        simulator_class,
        config,
        n_simulations,
        output_file,
        nuisance_pars,
        cyclical_pars,
        flipped_pars,
        prior_file,
    )


# ── save_data ─────────────────────────────────────────────────────────────────
@cli.command(
    "save_data",
    short_help="Save observed data bins from the simulator to parquet.",
)
@apply_options(_SIMULATOR_OPTIONS)
def save_data(
    simulator_module: str,
    simulator_class: str,
    config: Path,
    output_file: Path,
    nuisance_pars: list[str],
    cyclical_pars: list[str],
    flipped_pars: list[str],
) -> None:
    save_data_module(
        simulator_module,
        simulator_class,
        config,
        output_file,
        nuisance_pars,
        cyclical_pars,
        flipped_pars,
    )


# ── train ─────────────────────────────────────────────────────────────────────
@cli.command("train", short_help="Train the NPE density estimator.")
# I/O
@optgroup.group("Input / Output")
@optgroup.option(
    "--save_file",
    "-s",
    required=True,
    help="Base path for saving the model. Checkpoints are written relative to this.",
)
@optgroup.option(
    "--dataset",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="Path to folder of .feather simulation files.",
)
@optgroup.option(
    "--prior_path",
    "-r",
    required=True,
    help="Path to a saved Prior .pkl file.",
)
@optgroup.option(
    "--nuisance_pars",
    "-p",
    multiple=True,
    help="Parameter name patterns (fnmatch-style) to exclude from training.",
)
# Model architecture
@optgroup.group("Model Architecture")
@optgroup.option(
    "--model",
    default="zuko_maf",
    show_default=True,
    help="Density estimator architecture: 'maf' (Masked Autoregressive Flow) or 'nse' (Neural Spline Flow).",
)
@optgroup.option(
    "--hidden",
    default=64,
    type=int,
    show_default=True,
    help="Number of hidden units per layer.",
)
@optgroup.option(
    "--num_blocks",
    default=2,
    type=int,
    show_default=True,
    help="Number of residual blocks.",
)
@optgroup.option(
    "--dropout",
    default=0.2,
    type=float,
    show_default=True,
    help="Dropout probability applied during training.",
)
@optgroup.option(
    "--transforms",
    default=5,
    type=int,
    show_default=True,
    help="Number of autoregressive transforms (MAF only).",
)
@optgroup.option(
    "--num_bins",
    default=10,
    type=int,
    show_default=True,
    help="Number of spline bins (NSF only).",
)
# Training
@optgroup.group("Training")
@optgroup.option(
    "--batch_size",
    default=2048,
    type=int,
    show_default=True,
    help="Number of samples per training batch.",
)
@optgroup.option(
    "--max_epochs",
    default=int(1e5),
    type=int,
    show_default=True,
    help="Maximum number of training epochs.",
)
@optgroup.option(
    "--learning_rate",
    default=5e-4,
    type=float,
    show_default=True,
    help="Initial learning rate for the Adam optimiser.",
)
@optgroup.option(
    "--ema_alpha",
    default=0.01,
    type=float,
    show_default=True,
    help="Smoothing factor for the EMA of validation loss used in early stopping.",
)
@optgroup.option(
    "--stop_after_epochs",
    default=100,
    type=int,
    show_default=True,
    help="Stop if EMA validation loss has not improved for this many epochs.",
)
@optgroup.option(
    "--validation_fraction",
    default=0.1,
    type=float,
    show_default=True,
    help="Fraction of data held out for validation.",
)
@optgroup.option(
    "--scheduler_patience",
    default=50,
    type=int,
    show_default=True,
    help="Epochs without improvement before ReduceLROnPlateau halves the learning rate.",
)
# Performance
@optgroup.group("Performance")
@optgroup.option(
    "--num_workers",
    default=1,
    type=int,
    show_default=True,
    help="Number of DataLoader worker processes.",
)
@optgroup.option(
    "--use_amp",
    is_flag=True,
    default=False,
    help="Enable automatic mixed precision (AMP). May not improve performance on all hardware.",
)
@optgroup.option(
    "--compile_model",
    is_flag=True,
    default=False,
    help="Compile with torch.compile. Reduces per-step time on supported hardware but increases startup time.",
)
@optgroup.option(
    "--prune_model",
    type=float,
    default=None,
    help="Prune the model. This reduces the number of nodes dynamically but may compromise accuracy.",
)
# Checkpointing & logging
@optgroup.group("Checkpointing & Logging")
@optgroup.option(
    "--autosave_every",
    default=1,
    type=int,
    show_default=True,
    help="Save a periodic checkpoint every N epochs (best-model saves are always written).",
)
@optgroup.option(
    "--resume_checkpoint",
    type=click.Path(exists=True),
    default=None,
    help="Path to a checkpoint file to resume training from.",
)
@optgroup.option(
    "--print_interval",
    default=1,
    type=int,
    show_default=True,
    help="Log training progress every N epochs.",
)
@optgroup.option(
    "--tensorboard_dir",
    default=None,
    help="Directory for TensorBoard event files. Omit to disable TensorBoard logging.",
)
@optgroup.option(
    "--show_progress",
    is_flag=True,
    default=False,
    help="Show two-level fit/epoch progress bars (works in CLI and Jupyter).",
)
def train(
    save_file: Path,
    prior_path: Path,
    dataset: Path,
    nuisance_pars: list[str],
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
) -> None:
    """Train a Neural Posterior Estimation (NPE) density estimator.

    Loads simulations from ``--dataset``, builds an NPE model with the
    specified architecture, and trains it with a custom loop featuring
    linear warm-up, ReduceLROnPlateau scheduling, EMA-based early stopping,
    and periodic checkpointing.

    The full model configuration (architecture + hyperparameters) is embedded
    in every checkpoint, so ``inference`` requires no architecture flags.

    Example::

        mach3sbi train \\
            -r prior.pkl -d sims/ -s models/run.pt \\
            --model maf --hidden 128 --transforms 8 \\
            --max_epochs 50000 --stop_after_epochs 200
    """

    train_module(
        save_file,
        prior_path,
        dataset,
        nuisance_pars,
        model,
        hidden,
        dropout,
        num_blocks,
        transforms,
        num_bins,
        batch_size,
        max_epochs,
        ema_alpha,
        learning_rate,
        stop_after_epochs,
        validation_fraction,
        num_workers,
        autosave_every,
        resume_checkpoint,
        use_amp,
        print_interval,
        tensorboard_dir,
        scheduler_patience,
        show_progress,
        compile_model,
        prune_model,
    )


# ── inference ─────────────────────────────────────────────────────────────────
@cli.command(short_help="Sample the posterior given observed data.")
# I/O
@optgroup.group("Input / Output")
@optgroup.option(
    "--posterior",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Path to a saved density estimator checkpoint (.pt / .ckpt). "
    "The model architecture is read directly from the checkpoint — "
    "no architecture flags are needed.",
)
@optgroup.option(
    "--observed_data_file",
    "-o",
    type=click.Path(exists=True),
    required=True,
    help="Path to the observed data parquet file (produced by save_data).",
)
@optgroup.option(
    "--save_file",
    "-s",
    required=True,
    help="Where to save the posterior samples (parquet).",
)
@optgroup.option(
    "--prior_path",
    "-r",
    required=True,
    help="Path to a saved Prior .pkl file.",
)
@optgroup.option(
    "--nuisance_pars",
    "-p",
    multiple=True,
    help="Parameter name patterns to exclude (must match those used during training).",
)
# Sampling
@optgroup.group("Sampling")
@optgroup.option(
    "--n_samples",
    "-n",
    required=True,
    type=int,
    help="Number of posterior samples to draw.",
)
def inference(
    posterior: Path,
    prior_path: Path,
    save_file: Path,
    n_samples: int,
    observed_data_file: Path,
    nuisance_pars: list[str],
) -> None:
    """Sample the posterior distribution conditioned on observed data.

    Loads a trained density estimator checkpoint, reads the model architecture
    directly from it (no ``--model`` / ``--hidden`` / etc. flags required),
    conditions the posterior on the observed data vector, draws
    ``--n_samples`` samples, and writes them as a parquet file with one column
    per parameter.

    Example::

        mach3sbi inference \\
            -i models/best.pt -r prior.pkl \\
            -n 100000 -o observed.parquet -s samples.parquet
    """
    inference_module(
        posterior,
        prior_path,
        save_file,
        n_samples,
        observed_data_file,
        nuisance_pars,
    )


# ── Importance sampling ─────────────────────────────────────────────────────────────────
@cli.command(short_help="Importance sample the posterior given observed data.")
@optgroup.group("Input / Output")
@optgroup.option(
    "--posterior",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Path to a saved density estimator checkpoint (.pt / .ckpt). "
    "The model architecture is read directly from the checkpoint — "
    "no architecture flags are needed.",
)
# Sampling
@optgroup.group("Sampling")
@optgroup.option(
    "--n_samples",
    "-n",
    required=True,
    type=int,
    help="Number of posterior samples to draw.",
)
@optgroup.option(
    "--oversampling_factor",
    required=False,
    default=5,
    type=int,
    help="Oversampling rate.",
)
@optgroup.option(
    "--max-sampling-batch",
    required=False,
    default=10_000,
    type=int,
    help="Max sampling batch size.",
)
@apply_options(_SIMULATOR_OPTIONS)
def importance_sample(
    simulator_module: str,
    simulator_class: str,
    config: Path,
    output_file: Path,
    n_samples: int,
    oversampling_factor: int,
    max_sampling_batch: int,
    posterior: Path,
    nuisance_pars: list[str],
    cyclical_pars: list[str],
    flipped_pars: list[str],
):
    importance_sample_module(
        simulator_module,
        simulator_class,
        config,
        output_file,
        n_samples,
        oversampling_factor,
        max_sampling_batch,
        posterior,
        nuisance_pars,
        cyclical_pars,
        flipped_pars,
    )


@cli.command(short_help="Run model diagnostics")
@apply_options(_SIMULATOR_OPTIONS)
@optgroup.group("Input / Output")
@optgroup.option(
    "--posterior",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Path to a saved density estimator checkpoint (.pt / .ckpt). "
    "The model architecture is read directly from the checkpoint — "
    "no architecture flags are needed.",
)
@optgroup.group("Parameters")
@optgroup.option(
    "--nuisance_pars",
    "-p",
    multiple=True,
)
@optgroup.option(
    "--cyclical_pars",
    "-cy",
    multiple=True,
)
@optgroup.group("Diagnostic Types")
@optgroup.option(
    "--make_sbc_rank",
    is_flag=True,
    default=False,
)
@optgroup.option(
    "--make_expected_coverage",
    is_flag=True,
    default=False,
)
@optgroup.option(
    "--make_tarp",
    is_flag=True,
    default=False,
)
@optgroup.option(
    "--make_logl_comp",
    is_flag=True,
    default=False,
)
@optgroup.option("--n_prior_samples", "-n", type=int, default=200)
@optgroup.option("--n_posterior_samples", type=int, default=1000)
def diagnostics(
    simulator_module: str,
    simulator_class: str,
    config: Path,
    posterior: Path,
    output_file: Path,
    nuisance_pars: list[str],
    cyclical_pars: list[str],
    flipped_pars: list[str],
    # Plot opts.
    make_sbc_rank: bool,
    make_expected_coverage: bool,
    make_tarp: bool,
    make_logl_comp: bool,
    n_prior_samples: int,
    n_posterior_samples: int,
) -> None:

    diagnostics_module(
        simulator_module,
        simulator_class,
        config,
        posterior,
        output_file,
        nuisance_pars,
        cyclical_pars,
        flipped_pars,
        make_sbc_rank,
        make_expected_coverage,
        make_tarp,
        make_logl_comp,
        n_prior_samples,
        n_posterior_samples,
    )
