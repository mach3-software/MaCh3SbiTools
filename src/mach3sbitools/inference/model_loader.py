"""
HW: Code to perform inference
"""

from pathlib import Path, PosixPath, WindowsPath
from typing import cast

import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)

from mach3sbitools.utils import (
    PosteriorConfig,
    TrainingConfig,
    get_logger,
)

from .inference_utils import ModelState

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


def _strip_compiled_prefix(state_dict: dict) -> dict:
    """Remove the ``_orig_mod.`` prefix that torch.compile adds."""
    if any(k.startswith("_orig_mod.") for k in state_dict):
        logger.debug("Stripped _orig_mod. prefix from compiled model state dict")
        return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    return state_dict


class ModelLoader:
    def __init__(self, model_path: Path):
        """
        Loads a model from checkpoint
        :param model_path: Path to a checkpoint
        """

        if not model_path.is_file():
            raise FileNotFoundError(f"Cannot find {model_path}")

        self.model_path = model_path

        # We load the checkpoint dict
        self._checkpoint_dict = self._load_checkpoint()
        get_logger().info("Loaded %s", model_path)

    def _load_checkpoint(self) -> ModelState:
        return cast(
            ModelState,
            torch.load(self.model_path, map_location="cpu", weights_only=False),
        )

    def load_into(self, model: torch.nn.Module) -> None:
        """
        Load this checkpoint's weights into ``model`` in place.

        Works whether ``model`` is a plain module or has already been
        FSDP2-sharded (e.g. via ``configure_model`` under
        ``ModelParallelStrategy``) — sharded parameters are correctly
        re-sharded from the full state dict rather than requiring an
        exact DTensor-for-DTensor match.

        :param model: Target module to load weights into.
        """
        is_sharded = any(type(p).__name__ == "DTensor" for p in model.parameters())
        if is_sharded:
            options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
            set_model_state_dict(model, self.state_dict, options=options)
        else:
            model.load_state_dict(self.state_dict)

    @property
    def x_dim(self) -> int:
        """Get x-dimension from saved checkpoint"""
        return self._checkpoint_dict["x_dim"]

    @property
    def theta_dim(self) -> int:
        """Get theta-dimension from saved checkpoint"""
        return self._checkpoint_dict["theta_dim"]

    @property
    def model_config(self) -> PosteriorConfig:
        """Get config from saved checkpoint"""
        return self._checkpoint_dict["model_config"]

    @property
    def epoch(self) -> int:
        """Get epoch from saved checkpoint"""
        return self._checkpoint_dict["epoch"]

    @property
    def state_dict(self):
        return _strip_compiled_prefix(self._checkpoint_dict["model_state"])

    @property
    def x_compressor(self) -> dict | None:
        return self._checkpoint_dict["x_compressor"]

    @property
    def theta_compressor(self) -> dict | None:
        return self._checkpoint_dict["theta_compressor"]
