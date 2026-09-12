"""
Data containers for prior parameters.
"""

from dataclasses import dataclass

import numpy as np
import torch

from mach3sbitools.utils import get_device, to_tensor


@dataclass(eq=False, repr=False)
class PriorData(torch.nn.Module):
    """
    Dataclass holding the raw prior parameter arrays.

    Subclasses :class:`torch.nn.Module` so that tensors can be moved to a
    device via standard PyTorch mechanisms.

    :param parameter_names: Array of parameter name strings, shape ``(n_params,)``.
    :param nominals: Nominal (mean) values, shape ``(n_params,)``.
    :param covariance_matrix: Full covariance matrix, shape ``(n_params, n_params)``.
    :param lower_bounds: Hard lower bounds, shape ``(n_params,)``.
    :param upper_bounds: Hard upper bounds, shape ``(n_params,)``.
    """

    parameter_names: np.ndarray
    nominals: torch.Tensor
    covariance_matrix: torch.Tensor
    lower_bounds: torch.Tensor
    upper_bounds: torch.Tensor

    def __post_init__(self) -> None:
        """
        Initialise the Module base and coerce every field to a tensor.

        Fields may arrive as numpy arrays or lists, so this converts rather
        than only moving.
        """
        super().__init__()
        device = get_device()
        self.nominals = to_tensor(self.nominals, device)
        self.covariance_matrix = to_tensor(self.covariance_matrix, device)
        self.lower_bounds = to_tensor(self.lower_bounds, device)
        self.upper_bounds = to_tensor(self.upper_bounds, device)

    def to(self, device: torch.device | str) -> "PriorData":  # type: ignore[override]
        """
        Move all tensor fields to *device* in-place.

        Overrides :meth:`torch.nn.Module.to`, which only recurses into
        registered parameters/buffers/submodules and would otherwise leave
        these plain tensor attributes untouched.

        :param device: Target PyTorch device.
        :returns: ``self``, for chaining.
        """
        self.nominals = self.nominals.to(device)
        self.covariance_matrix = self.covariance_matrix.to(device)
        self.lower_bounds = self.lower_bounds.to(device)
        self.upper_bounds = self.upper_bounds.to(device)
        return self

    def __getitem__(self, mask: torch.Tensor) -> "PriorData":
        """
        Return a masked subset of the prior data.

        :param mask: Boolean tensor of shape ``(n_params,)``.
        :returns: New :class:`PriorData` containing only the selected parameters.
        """
        np_mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        # Match the mask to wherever this instance's data actually lives, not
        # to the globally detected device: this PriorData may have been moved
        # on its own.
        tensor_mask = to_tensor(mask, device=self.nominals.device)

        return PriorData(
            parameter_names=self.parameter_names[np_mask],
            nominals=self.nominals[tensor_mask],
            covariance_matrix=self.covariance_matrix[tensor_mask][:, tensor_mask],
            lower_bounds=self.lower_bounds[tensor_mask],
            upper_bounds=self.upper_bounds[tensor_mask],
        )
