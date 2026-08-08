"""
Feather file I/O utilities for simulation data.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

import numpy as np
import pyarrow as pa
import torch
from pyarrow import Table, feather

from mach3sbitools.types import SimulatorData, SimulatorDataGrouped


class FeatherOutput(TypedDict):
    """Schema for feather files written by :func:`to_feather`."""

    x: SimulatorData
    theta: SimulatorData


def _list_column_to_numpy(
    column: pa.ChunkedArray | pa.Array, dtype: type = np.float32
) -> np.ndarray:
    """
    Convert a fixed-width Arrow list-typed column to a 2D numpy array.

    ``theta``/``x`` columns are written as Arrow ``list<double>`` columns
    (every row's list is the same length, but Arrow itself doesn't know
    that). The naive way to read one back is ``column.to_list()`` /
    ``.to_pylist()``, which round-trips every element through a Python
    float object before ``np.array()`` re-packs them — for a handful of
    columns that's fine, but at ~1000 columns it means hundreds of millions
    of transient Python objects per shard (tens of seconds and several GB
    of peak RAM just to decode one file).

    Instead: grab the list column's flat child array directly (all rows'
    values concatenated, no per-row Python objects), convert *that* to
    numpy in one call, and reshape using the row count — order is
    preserved because Arrow's list child array is exactly the row-major
    concatenation of each row's values. Single-chunk columns (the normal
    case — one write produces one chunk) skip ``combine_chunks()``
    entirely. Measured ~30x faster than ``.to_list()`` at 1000 columns.

    :param column: The Arrow list-typed column to convert.
    :param dtype: Output numpy dtype.
    :returns: A ``(n_rows, n_cols)`` numpy array.
    """
    n_rows = len(column)
    if isinstance(column, pa.ChunkedArray):
        chunk = column.chunk(0) if column.num_chunks == 1 else column.combine_chunks()
    else:
        chunk = column

    # NB: use .flatten(), not .values — for a *sliced* list array (as used
    # by iter_feather_chunks's Table.slice), .values returns the full
    # underlying child buffer ignoring the slice, while .flatten() correctly
    # returns only the values belonging to this array's own rows.
    flat = chunk.flatten().to_numpy(zero_copy_only=False)
    n_cols = flat.shape[0] // n_rows
    return flat.reshape(n_rows, n_cols).astype(dtype, copy=False)


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

    table = feather.read_table(str(file_name), memory_map=True)
    theta = _list_column_to_numpy(table["theta"], dtype=np.float32)
    x = _list_column_to_numpy(table["x"], dtype=np.float32)

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


def count_feather_rows(file_name: Path) -> int:
    """
    Count the rows in a feather file without materialising any column data.

    Feather (v2) is the Arrow IPC file format, so the number of rows per
    record batch is available from batch metadata alone. This is what makes
    it cheap to build a row-level index over a large number of shards up
    front (e.g. for :class:`~mach3sbitools.data_loaders.StreamingFeatherDataset`)
    without paying the cost of decoding ``theta``/``x`` for every file.

    :param file_name: Path to the ``.feather`` file.
    :returns: Total number of rows in the file.
    :raises FileNotFoundError: If *file_name* does not exist.
    """
    if not isinstance(file_name, Path):
        file_name = Path(file_name)

    if not file_name.exists():
        raise FileNotFoundError(file_name)

    with pa.memory_map(str(file_name), "r") as mmap:
        reader = pa.ipc.open_file(mmap)
        return sum(
            reader.get_batch(i).num_rows for i in range(reader.num_record_batches)
        )


def iter_feather_chunks(
    file_name: Path, chunk_rows: int
) -> Iterator[SimulatorDataGrouped]:
    """
    Stream a (possibly very large) feather file in fixed-size row chunks.

    Opens the file memory-mapped so the OS pages in only the bytes touched
    by each ``Table.slice`` — the file is never fully materialised in RAM,
    which is what lets this handle a single monolithic shard (e.g. one
    ``mach3sbi simulate`` run that wrote all ``N`` simulations to one file)
    with a bounded, predictable memory footprint. Used by the ``reshard``
    CLI command to split such files into many smaller shards.

    :param file_name: Path to the ``.feather`` file to stream.
    :param chunk_rows: Number of rows to read into memory at a time.
    :yields: ``(theta, x)`` float32 numpy array pairs, one per chunk.
    :raises FileNotFoundError: If *file_name* does not exist.
    """
    if not isinstance(file_name, Path):
        file_name = Path(file_name)

    if not file_name.exists():
        raise FileNotFoundError(file_name)

    table = feather.read_table(str(file_name), memory_map=True)
    n_rows = table.num_rows

    for offset in range(0, n_rows, chunk_rows):
        length = min(chunk_rows, n_rows - offset)
        chunk = table.slice(offset, length)
        theta = _list_column_to_numpy(chunk["theta"], dtype=np.float32)
        x = _list_column_to_numpy(chunk["x"], dtype=np.float32)
        yield theta, x
