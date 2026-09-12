from typing import TypedDict

import torch
from lightning.pytorch.strategies import (
    DDPStrategy,
    ModelParallelStrategy,
    Strategy,
)

from mach3sbitools.utils import PosteriorConfig, get_device, get_logger

logger = get_logger()


class ModelState(TypedDict):
    """Schema of the checkpoint dict written by ``SBILightningModule``."""

    model_state: dict
    model_config: PosteriorConfig
    epoch: int
    theta_dim: int
    theta_compressor: dict | None
    x_dim: int
    x_compressor: dict | None


def select_accelerator_and_strategy(
    use_model_parallel: bool = False,
) -> tuple[str, str | Strategy]:
    """
    Pick the Lightning accelerator and distribution strategy for this host.

    The device comes from :func:`~mach3sbitools.utils.get_device`, so the
    trainer cannot disagree with the rest of the package about where tensors
    live.

    Density estimators here are small, so distribution is only worth its
    communication cost when there is genuinely more than one GPU: a single
    GPU gets ``"auto"`` (no process group at all) rather than DDP-of-one.

    :param use_model_parallel: Shard the model across GPUs with FSDP2 rather
        than replicating it with DDP. Ignored without multiple CUDA devices.
    :returns: Tuple of ``(accelerator, strategy)`` for ``lightning.Trainer``.
    """
    device = get_device()

    if device.type != "cuda":
        return device.type, "auto"

    n_devices = torch.cuda.device_count()
    if n_devices <= 1:
        return "gpu", "auto"

    if use_model_parallel:
        return "gpu", ModelParallelStrategy()

    return "gpu", DDPStrategy(
        # The flow uses every parameter on every step, so DDP does not need to
        # trace the graph looking for unused ones.
        find_unused_parameters=False,
        # Let gradients alias the reduction buckets instead of being copied in.
        gradient_as_bucket_view=True,
    )


def select_model_kwargs(config: PosteriorConfig) -> dict:
    """
    Filter *config* down to the kwargs the chosen flow family accepts.

    Zuko-backed flows reject MAF/MLP concepts such as ``num_blocks``, so
    passing the full config surface would raise a ``TypeError`` inside sbi.
    Unknown model names pass everything through and let sbi complain.

    :param config: Requested architecture settings.
    :returns: The subset of kwargs ``posterior_nn`` will accept.
    """

    model_factory: dict[str, set[str]] = {
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
    all_kwargs = {
        "hidden_features": config.hidden_features,
        "num_transforms": config.num_transforms,
        "dropout_probability": config.dropout_probability,
        "num_blocks": config.num_blocks,
        "num_bins": config.num_bins,
    }

    accepted = model_factory.get(config.model.lower(), set(all_kwargs.keys()))
    filtered = {k: v for k, v in all_kwargs.items() if k in accepted}

    dropped = set(all_kwargs) - set(filtered)
    if dropped:
        logger.debug(
            "Model '%s' does not accept %s; these kwargs were dropped from the posterior_nn call.",
            config.model,
            sorted(dropped),
        )
    return filtered
