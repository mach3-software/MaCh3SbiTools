"""
Merge feather shards into memmap-backed .npy files (theta.npy, x.npy).

Same two-pass structure as the HDF5 version (count rows, then peek dims,
then stream-copy), but writes straight to np.lib.format.open_memmap instead
of h5py. Output is two files in output_dir: theta.npy and x.npy, directly
usable with np.load(path, mmap_mode="r") / MemmapPairDataset - no separate
conversion step needed.
"""

from pathlib import Path

import numpy as np
from tqdm.rich import tqdm
from tqdm import TqdmExperimentalWarning
import warnings

from mach3sbitools.utils import from_feather, get_logger, peek_num_rows

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

# Set to np.float32 to downcast during merge and roughly halve output size
# vs the source float64 feather data. Set to None to keep source dtype.
FORCE_DTYPE = None


def _truncate_npy(path: Path, new_n_rows: int, chunk_rows: int = 100_000):
    """
    Rewrite a memmap-backed .npy file so it only contains its first
    new_n_rows rows, dropping any trailing unused/uninitialized rows.

    We can't just slice the file in place because the .npy header (which
    encodes the shape) is padded/aligned to 64 bytes, and that padding can
    shift width depending on the digit count of the shape - so we rewrite
    via open_memmap with the correct shape and stream-copy the data across
    in chunks to avoid loading the whole array into memory.
    """
    arr = np.load(path, mmap_mode="r")
    if new_n_rows == arr.shape[0]:
        return

    new_shape = (new_n_rows,) + arr.shape[1:]
    tmp_path = path.with_suffix(".npy.tmp")
    new_arr = np.lib.format.open_memmap(
        tmp_path, mode="w+", dtype=arr.dtype, shape=new_shape
    )
    for start in range(0, new_n_rows, chunk_rows):
        end = min(start + chunk_rows, new_n_rows)
        new_arr[start:end] = arr[start:end]
    new_arr.flush()

    del arr, new_arr  # release mmaps before replacing the file
    tmp_path.replace(path)


def merge_shards_module(simulation_dir: Path, output_dir: Path):
    """
    Merge a folder of feather shard files into memmap-backed theta.npy / x.npy
    in output_dir.
    """
    theta_path = output_dir / "theta.npy"
    x_path = output_dir / "x.npy"

    for p in (theta_path, x_path):
        if p.exists():
            raise FileExistsError(
                f"{p} already exists please rename or save to another output dir"
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    sims_files = list(simulation_dir.glob("*feather"))
    if not sims_files:
        raise FileNotFoundError(f"Cannot find any .feather files in {simulation_dir}")

    get_logger().info(
        "Merging %d files in %s -> %s", len(sims_files), simulation_dir, output_dir
    )

    n_rows = 0
    for shard in tqdm(sims_files, desc="Counting number of rows"):
        n_rows += peek_num_rows(shard)

    # We now peek the first entry
    t_test, x_test = from_feather(sims_files[0])

    # We get the theta and x_dim
    t_dim = len(t_test[0])
    x_dim = len(x_test[0])

    theta_dtype = FORCE_DTYPE or t_test.dtype
    x_dtype = FORCE_DTYPE or x_test.dtype

    del t_test, x_test

    # Create the memmap-backed .npy files up front, sized for the full
    # merged dataset (pre-filter) - same role as h5py.File.create_dataset
    # before. Since rows get dropped by the filter below, we may end up
    # writing fewer than n_rows rows; the tail is trimmed off at the end.
    theta_out = np.lib.format.open_memmap(
        theta_path, mode="w+", dtype=theta_dtype, shape=(n_rows, t_dim)
    )
    x_out = np.lib.format.open_memmap(
        x_path, mode="w+", dtype=x_dtype, shape=(n_rows, x_dim)
    )

    desc_str = f"Adding sims to {output_dir} | current file: "

    offset = 0
    total_filtered = 0

    for shard in (pbar := tqdm(sims_files, desc=desc_str + str(sims_files[0]))):
        pbar.set_description(desc_str + str(shard))

        t, x = from_feather(shard)

        before = len(t)

        # HACK
        # t_filter = np.where(t[:, -2] > 0)

        # t = t[t_filter]
        # x = x[t_filter]

        # total_filtered += before - len(t)

        theta_out[offset : offset + len(t)] = t
        x_out[offset : offset + len(x)] = x
        offset += len(t)

    theta_out.flush()
    x_out.flush()
    del theta_out, x_out  # release mmaps so the files can be truncated/replaced

    if offset < n_rows:
        get_logger().info(
            "Trimming %d unused/uninitialized rows from output arrays",
            n_rows - offset,
        )
        _truncate_npy(theta_path, offset)
        _truncate_npy(x_path, offset)

    get_logger().info(
        "Finished merge. Filtered out %d/%d entries", total_filtered, n_rows
    )