"""
Feather file I/O utilities for simulation data.

Files are written uncompressed so they can be memory-mapped and read
lazily; compressed feather files must be fully decompressed on open.
"""

from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import pyarrow as pa
from pyarrow import Table, feather, ipc, memory_map

from mach3sbitools.types import SimulatorData, SimulatorDataGrouped


class FeatherOutput(TypedDict):
    """Schema for feather files written by :func:`to_feather`."""

    x: SimulatorData
    theta: SimulatorData


def from_feather(file_name: Path) -> SimulatorDataGrouped:
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

    with memory_map(str(file_name), "r") as source:
        table = ipc.open_file(source).read_all()
        theta = _column_to_2d(table["theta"]).astype(np.float32, copy=True)
        x = _column_to_2d(table["x"]).astype(np.float32, copy=True)

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
    feather.write_feather(param_table, str(file_name), compression="uncompressed")


def peek_num_rows(file_name: Path) -> int:
    """Read a feather file's row count from its footer, without loading data."""
    if not isinstance(file_name, Path):
        file_name = Path(file_name)

    with memory_map(str(file_name), "r") as source:
        return cast(int, ipc.open_file(source).read_all().num_rows)


def _column_to_2d(column: pa.ChunkedArray) -> np.ndarray:
    """Convert a fixed-width list column into a 2D numpy array (near zero-copy)."""
    arr = column.combine_chunks()
    n_rows = len(arr)
    flat = arr.flatten().to_numpy(zero_copy_only=False)
    n_features = flat.shape[0] // n_rows
    return flat.reshape(n_rows, n_features)


class FeatherFileHandle:
    """
    A memory-mapped feather file, exposing ``theta``/``x`` as numpy arrays
    backed by the mmap. Requires the file to have been written with
    ``compression="uncompressed"``.
    """

    __slots__ = ("_source", "path", "theta", "x")

    def __init__(self, path: Path):
        self.path = Path(path)
        self._source = memory_map(str(self.path), "r")
        table = ipc.open_file(self._source).read_all()
        self.theta = _column_to_2d(table["theta"])
        self.x = _column_to_2d(table["x"])

    @property
    def num_rows(self) -> int:
        return cast(int, self.theta.shape[0])

    def close(self) -> None:
        self._source.close()
