"""
Feather file I/O utilities for simulation data.
"""

from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from pyarrow import Table, feather

from mach3sbitools.types import SimulatorData, SimulatorDataGrouped


class FeatherOutput(TypedDict):
    """Schema for feather files written by :func:`to_feather`."""

    x: SimulatorData
    theta: SimulatorData


# def filter_nuisance(
#     parameter_names: list[str], nuisance_pars: list[str], theta: SimulatorData
# ) -> SimulatorData:
#     """
#     Remove nuisance parameters from a theta array by name pattern.

#     :param parameter_names: Ordered parameter names, length must match
#         ``theta.shape[1]``.
#     :param nuisance_pars: fnmatch patterns for parameters to exclude
#         (e.g. ``["syst_*"]``).
#     :param theta: Parameter array of shape ``(n_samples, n_params)``.
#     :returns: Filtered array with nuisance columns removed.
#     :raises ValueError: If ``len(parameter_names) != theta.shape[1]``.
#     """

#     if nuisance_pars is None:
#         if len(theta[0]) != len(parameter_names):
#             raise ValueError("Parameter names and theta must have same length")
#         return theta

#     param_filter = np.array(
#         [
#             not any(fnmatch(param, nuis) for nuis in nuisance_pars)
#             for param in parameter_names
#         ],
#         dtype=bool,
#     )
#     print(param_filter)

#     return theta[:, param_filter].copy()


def from_feather(
    file_name: Path, nuisance_filter: torch.Tensor | np.ndarray | None = None
) -> SimulatorDataGrouped:
    """
    Load a ``(theta, x)`` pair from a feather file.

    :param file_name: Path to the ``.feather`` file.
    :param nuisance_filter: fnmatch patterns for parameters to exclude from
        *theta*. ``None`` returns all parameters.
    :returns: Tuple of ``(theta, x)`` as ``float32`` numpy arrays.
    :raises FileNotFoundError: If *file_name* does not exist.
    """
    if not isinstance(file_name, Path):
        file_name = Path(file_name)

    if not file_name.exists():
        raise FileNotFoundError(file_name)

    table = feather.read_feather(str(file_name))
    theta = np.array(table["theta"].to_list(), dtype=np.float32)
    x = np.array(table["x"].to_list(), dtype=np.float32)

    # HW : Don't love the nesting, but oh well
    if nuisance_filter is not None:
        if isinstance(nuisance_filter, torch.Tensor):
            nuisance_filter = nuisance_filter.to("cpu").numpy()

        theta = theta[:, nuisance_filter]

    return theta, x


def to_feather(
    file_name: Path,
    theta_values: SimulatorData,
    x_values: SimulatorData,
) -> None:
    """
    Write a ``(theta, x)`` pair to a feather file.

    :param file_name: Destination path. Must end in ``.feather``.
    :param theta_values: Parameter array of shape ``(n_samples, n_params)``.
    :param x_values: Observable array of shape ``(n_samples, n_bins)``.
    :raises ValueError: If *file_name* does not have a ``.feather`` suffix.
    """
    if not isinstance(file_name, Path):
        file_name = Path(file_name)

    if file_name.suffix != ".feather":
        raise ValueError("Must store outputs files with the *.feather extension")

    param_dict: FeatherOutput = {
        "x": x_values.tolist(),
        "theta": theta_values.tolist(),
    }
    param_table = Table.from_pydict(param_dict)
    file_name.parent.mkdir(parents=True, exist_ok=True)
    feather.write_feather(param_table, str(file_name))
