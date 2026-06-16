"""
Configuration dataclasses for model architecture and training.
"""

from dataclasses import dataclass
from pathlib import Path
from pydantic import validate_call

@validate_call
@dataclass
class TrainingConfig:
    """
    Configuration for the SBI training loop.

    :param save_path: Directory to write model checkpoints. ``None`` disables
        checkpointing.
    :param batch_size: Number of samples per training batch.
    :param learning_rate: Initial learning rate for the Adam optimiser.
    :param max_epochs: Hard upper limit on training epochs.
    :param stop_after_epochs: Stop if the EMA validation loss has not improved
        for this many consecutive epochs.
    :param scheduler_patience: Epochs without improvement before
        :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau` halves the LR.
    :param validation_fraction: Fraction of data held out for validation.
    :param num_workers: Number of DataLoader worker processes.
    :param autosave_every: Save a periodic checkpoint every *N* epochs.
    :param resume_checkpoint: Path to a checkpoint to resume from.
    :param use_amp: Enable automatic mixed precision.
    :param print_interval: Log a training summary every *N* epochs.
    :param show_progress: Show the two-level fit/epoch progress bars.
        Works correctly in both CLI terminals and Jupyter notebooks.
        Set to ``False`` for non-interactive / CI environments.
    :param tensorboard_dir: Directory for TensorBoard event files.
        ``None`` disables TensorBoard logging.
    :param ema_alpha: EMA smoothing factor for validation loss used in early
        stopping. Smaller values are smoother.
    :param compile: Compile the model with ``torch.compile``.
    """

    save_path: Path | None = None
    batch_size: int = 2048
    learning_rate: float = 5e-4
    max_epochs: int = 500
    stop_after_epochs: int = 100
    scheduler_patience: int = 20
    validation_fraction: float = 0.1
    num_workers: int = 1
    autosave_every: int = 10
    resume_checkpoint: Path | None = None
    use_amp: bool = False
    print_interval: int = 10
    show_progress: bool = False
    tensorboard_dir: Path | None = None
    ema_alpha: float = 0.05
    compile: bool = False
    prune_model: float | None = None

@validate_call
@dataclass
class PosteriorConfig:
    """
    Configuration for the NPE density estimator architecture.

    :param model: ``"maf"`` (Masked Autoregressive Flow) or ``"nse"``
        (Neural Spline Flow).
    :param hidden_features: Number of hidden units per layer.
    :param num_transforms: Number of autoregressive transforms (MAF only).
    :param dropout_probability: Dropout probability during training.
    :param num_blocks: Number of residual blocks.
    :param num_bins: Number of spline bins (NSF only).
    """

    model: str = "maf"
    hidden_features: int = 128
    num_transforms: int = 6
    dropout_probability: float = 0.1
    num_blocks: int = 2
    num_bins: int = 10
