"""
Configuration dataclasses for model architecture and training.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    Configuration for the SBI training loop.

    :param save_path: Directory to write model checkpoints. ``None`` disables
        checkpointing.
    :param batch_size: Number of samples per training batch.
    :param learning_rate: Initial learning rate for the Adam optimiser.
    :param max_epochs: Hard upper limit on training epochs.
    :param stop_after_epochs: Stop if the validation loss has not improved for
        this many consecutive epochs.
    :param min_delta: Smallest change in validation loss that counts as an
        improvement for early stopping. Zero means any change resets the
        patience counter, so noise alone can keep a run alive; on a large
        dataset a small positive value (order 0.01 nats) is usually right.
    :param scheduler_patience: Epochs without improvement before
        :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau` halves the LR.
    :param validation_fraction: Fraction of data held out for validation.
    :param num_workers: Number of DataLoader worker processes. Raise this when
        the dataset is paged in from disk, so reads for the next batch overlap
        with compute on the current one.
    :param limit_train_batches: Cap the batches per training epoch. ``None``
        uses the whole training split. Setting it makes an "epoch" a fixed
        amount of work, so checkpointing, LR scheduling and early stopping
        operate on a sane cadence when the dataset is very large.
    :param limit_val_batches: Cap the batches per validation pass. ``None``
        validates the whole split, which on a large dataset spends real time
        re-deriving a loss estimate that a few hundred batches already pin
        down.
    :param autosave_every: Save a periodic checkpoint every *N* epochs.
    :param resume_checkpoint: Path to a checkpoint to resume from.
    :param use_amp: Enable automatic mixed precision.
    :param show_progress: Show the two-level fit/epoch progress bars.
        Works correctly in both CLI terminals and Jupyter notebooks.
        Set to ``False`` for non-interactive / CI environments.
    :param tensorboard_dir: Directory for TensorBoard event files.
        ``None`` disables TensorBoard logging.
    :param ema_alpha: EMA smoothing factor for the ``val/ema_loss``
        diagnostic. Smaller values are smoother. This is logged for
        inspection only — nothing makes decisions from it, because a smoothed
        metric lags the true optimum and would delay stopping and select a
        worse checkpoint.
    :param compile: Compile the model with ``torch.compile``.
    """

    save_path: Path | None = None
    batch_size: int = 2048
    learning_rate: float = 5e-4
    max_epochs: int = 500
    stop_after_epochs: int = 100
    min_delta: float = 0.0
    scheduler_patience: int = 20
    validation_fraction: float = 0.1
    num_workers: int = 4
    limit_train_batches: int | None = None
    limit_val_batches: int | None = None
    autosave_every: int = 10
    resume_checkpoint: Path | None = None
    use_amp: bool = False
    show_progress: bool = False
    tensorboard_dir: Path | None = None
    ema_alpha: float = 0.05
    compile: bool = False


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
