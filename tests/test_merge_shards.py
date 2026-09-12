"""
Tests for mach3sbitools.apps.merge_shards.
"""

import numpy as np
import pytest

from mach3sbitools.apps.merge_shards import merge_shards_module
from mach3sbitools.utils import to_feather

THETA_DIM = 6
X_DIM = 4


def _write_shards(shard_dir, rows_per_shard, theta_dim=THETA_DIM, x_dim=X_DIM):
    """
    Write one feather shard per entry in *rows_per_shard*.

    :param shard_dir: Directory to write into.
    :param rows_per_shard: Row count for each shard.
    :param theta_dim: Parameter width.
    :param x_dim: Observable width.
    :returns: The concatenated ``(theta, x)`` arrays that were written.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    thetas, xs = [], []
    for i, n_rows in enumerate(rows_per_shard):
        theta = rng.normal(size=(n_rows, theta_dim))
        x = rng.normal(size=(n_rows, x_dim))
        to_feather(shard_dir / f"shard_{i}.feather", theta, x)
        thetas.append(theta)
        xs.append(x)

    return np.concatenate(thetas), np.concatenate(xs)


class TestMergeShards:
    def test_writes_both_arrays(self, tmp_path):
        _write_shards(tmp_path / "shards", [10, 10, 10])
        out = tmp_path / "merged"
        merge_shards_module(tmp_path / "shards", out)

        assert (out / "theta.npy").is_file()
        assert (out / "x.npy").is_file()

    def test_row_count_and_widths(self, tmp_path):
        _write_shards(tmp_path / "shards", [7, 13, 5])
        out = tmp_path / "merged"
        merge_shards_module(tmp_path / "shards", out)

        theta = np.load(out / "theta.npy", mmap_mode="r")
        x = np.load(out / "x.npy", mmap_mode="r")

        assert theta.shape == (25, THETA_DIM)
        assert x.shape == (25, X_DIM)

    def test_values_survive_the_merge(self, tmp_path):
        theta_in, x_in = _write_shards(tmp_path / "shards", [4, 6])
        out = tmp_path / "merged"
        merge_shards_module(tmp_path / "shards", out)

        # Shards are globbed, so compare against a row-order-independent view.
        theta_out = np.sort(np.load(out / "theta.npy"), axis=0)
        x_out = np.sort(np.load(out / "x.npy"), axis=0)

        np.testing.assert_allclose(theta_out, np.sort(theta_in, axis=0), rtol=1e-5)
        np.testing.assert_allclose(x_out, np.sort(x_in, axis=0), rtol=1e-5)

    def test_memmap_loadable(self, tmp_path):
        """The output must be readable lazily — that is the point of .npy."""
        _write_shards(tmp_path / "shards", [10])
        out = tmp_path / "merged"
        merge_shards_module(tmp_path / "shards", out)

        theta = np.load(out / "theta.npy", mmap_mode="r")
        assert isinstance(theta, np.memmap)


class TestMergeShardsErrors:
    def test_raises_when_no_shards(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="feather"):
            merge_shards_module(empty, tmp_path / "merged")

    def test_refuses_to_overwrite(self, tmp_path):
        _write_shards(tmp_path / "shards", [5])
        out = tmp_path / "merged"
        merge_shards_module(tmp_path / "shards", out)

        with pytest.raises(FileExistsError):
            merge_shards_module(tmp_path / "shards", out)

    def test_rejects_inconsistent_shard_widths(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_shards(shard_dir, [5])
        rng = np.random.default_rng(1)
        to_feather(
            shard_dir / "shard_wide.feather",
            rng.normal(size=(5, THETA_DIM + 2)),
            rng.normal(size=(5, X_DIM)),
        )

        with pytest.raises(ValueError, match="expected theta"):
            merge_shards_module(shard_dir, tmp_path / "merged")
