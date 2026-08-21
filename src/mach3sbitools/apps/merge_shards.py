"""
HW: Merge feather shards into a single HDF5 file
"""

from pathlib import Path

import h5py
from tqdm.rich import tqdm

from mach3sbitools.utils import from_feather, get_logger, peek_num_rows


def merge_shards_module(simulation_dir: Path, output_file: Path):
    """
    Merge a folder of feather shard files into a single HDF5 output file
    """
    if output_file.exists():
        raise FileExistsError(
            f"{output_file} already exists please rename or save to another hdf5 file"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    sims_files = list(simulation_dir.glob("*feather"))
    if not sims_files:
        raise FileNotFoundError(f"Cannot find any .feather files in {simulation_dir}")

    get_logger().info(
        "Merging %d files in %s -> %s", len(sims_files), simulation_dir, output_file
    )

    n_rows = 0
    for shard in sims_files:
        n_rows += peek_num_rows(shard)

    # We now peak the first entry
    t_test, x_test = from_feather(sims_files[0])

    # We get the theta and x_dim
    t_dim = len(t_test[0])
    x_dim = len(x_test[0])

    del t_test, x_test

    with h5py.File(output_file) as f:
        # We now create separate x and theta dataset
        theta_dset = f.create_dataset("theta", (t_dim, n_rows))
        x_dset = f.create_dataset("x", (x_dim, n_rows))

        # Now we append the data set to the hdf5 file
        desc_str = f"Adding sims to {output_file} | current file: "

        offset = 0
        for shard in (pbar := tqdm(sims_files, desc=desc_str + str(sims_files[0]))):
            pbar.set_description(desc_str + str(shard))

            t, x = from_feather(shard)

            theta_dset[offset : offset + len(t)] = t
            x_dset[offset : offset + len(x)] = x

    get_logger().info("Finished merge")
