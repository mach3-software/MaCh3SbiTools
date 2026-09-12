"""
Device selection and tensor conversion.

A single :func:`get_device` is the one source of truth for where tensors
live. Everything that needs a device asks this module rather than probing
``torch.cuda`` itself, so the prior, the compressors, the dataset and the
Lightning trainer cannot end up disagreeing.
"""

import os
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

#: Environment variable that pins the device, e.g. ``MACH3SBI_DEVICE=cpu``.
DEVICE_ENV_VAR = "MACH3SBI_DEVICE"


class TensorConversionError(Exception):
    """Raised when an object cannot be converted to a :class:`torch.Tensor`."""


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    """
    Return the device every tensor in the package should live on.

    CUDA is used when available, otherwise CPU. Apple's MPS backend is
    deliberately *not* selected automatically: it has no float64 support and
    the priors are evaluated in double precision. Set
    ``MACH3SBI_DEVICE=mps`` to opt in anyway, or ``MACH3SBI_DEVICE=cpu`` to
    pin CPU on a machine that has a GPU.

    The result is cached for the life of the process — call
    :func:`reset_device_cache` if the environment changes underneath it.

    :returns: The selected :class:`torch.device`.
    """
    override = os.environ.get(DEVICE_ENV_VAR)
    if override:
        return torch.device(override)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def reset_device_cache() -> None:
    """Clear the :func:`get_device` cache, so the next call re-detects."""
    get_device.cache_clear()


def to_tensor(data, device: torch.device | str | None = None) -> torch.Tensor:
    """
    Convert an array-like object to a :class:`torch.Tensor` on *device*.

    Handles :class:`~pandas.DataFrame`, :class:`~numpy.ndarray`, lists and
    existing tensors, plus anything else :func:`torch.tensor` accepts.

    :param data: Input data to convert.
    :param device: Target device. Defaults to :func:`get_device`.
    :returns: Tensor on *device*.
    :raises TensorConversionError: If conversion fails.
    """
    device = torch.device(device) if device is not None else get_device()

    if isinstance(data, pd.DataFrame):
        return torch.tensor(data.values.astype(np.float32), device=device)
    if isinstance(data, np.ndarray):
        return torch.tensor(data.astype(np.float32), device=device)
    if isinstance(data, list):
        return torch.tensor(data, device=device)
    if isinstance(data, torch.Tensor):
        return data.clone().detach().to(device)

    try:
        return torch.tensor(data, device=device)
    except Exception as e:
        raise TensorConversionError(
            f"Cannot convert object of type {type(data)} to torch tensor"
        ) from e


class TorchDeviceHandler:
    """
    Legacy stand-in kept only so old pickles can be read.

    Priors saved before :func:`get_device` embedded one of these. Unpickling
    such a file needs the class to still resolve, even though
    :meth:`~mach3sbitools.simulator.priors.prior.Prior.__setstate__` discards
    the instance and re-detects the device.

    .. deprecated::
        Use :func:`get_device` and :func:`to_tensor`.
    """

    def __init__(self) -> None:
        """Record the detected device, matching the old constructor."""
        self._device = str(get_device())

    @property
    def device(self) -> str:
        """
        :returns: The detected device, as the string the old class returned.
        """
        return self._device

    def to_tensor(self, data) -> torch.Tensor:
        """
        Convert *data* to a tensor.

        :param data: Input data to convert.
        :returns: Tensor on the detected device.
        """
        return to_tensor(data)
