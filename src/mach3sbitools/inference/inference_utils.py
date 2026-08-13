from typing import TypedDict

import torch
from lightning.pytorch.strategies import ModelParallelStrategy

from mach3sbitools.utils import PosteriorConfig, get_logger

logger = get_logger()


class ModelState(TypedDict):
    model_state: dict
    model_config: PosteriorConfig
    epoch: int
    theta_dim: int
    theta_compressor: dict | None
    x_dim: int
    x_compressor: dict | None


def select_accelerator_and_strategy(
    use_model_parallel: bool = False,
) -> tuple[str, str | ModelParallelStrategy]:
    """Generates the model strategy + accelerator"""
    if torch.cuda.is_available():
        return "gpu", ModelParallelStrategy() if use_model_parallel else "ddp"
    if torch.backends.mps.is_available():
        return "mps", "auto"
    return "cpu", "auto"


def select_model_kwargs(config: PosteriorConfig) -> dict:
    """
    Internal method, bit hacky, gets model from kwargs
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
