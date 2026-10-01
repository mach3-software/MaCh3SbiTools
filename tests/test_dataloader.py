"""
Tests for mach3sbitools.data_loaders.TrainingDataset.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from mach3sbitools.apps.merge_shards import merge_shards_module
from mach3sbitools.data_loaders import TrainingDataset
from mach3sbitools.simulator import create_prior
from mach3sbitools.utils import to_feather

N_SHARDS = 3
ROWS_PER_SHARD = 50


@pytest.fixture(scope="module")
def indexed_shards(tmp_path_factory, test_consts) -> Path:
    """
    Feather shards where every value in a row equals its global row index,
    so it is easy to check which rows came back and in what order.
    """
    shard_dir: Path = tmp_path_factory.mktemp("indexed_shards")
    for i in range(N_SHARDS):
        rows = np.arange(i * ROWS_PER_SHARD, (i + 1) * ROWS_PER_SHARD, dtype=np.float64)
        theta = np.repeat(rows[:, None], test_consts.theta_dim, axis=1)
        x = np.repeat(rows[:, None], test_consts.x_dim, axis=1)
        to_feather(shard_dir / f"shard{i}.feather", theta, x)
    return shard_dir


@pytest.fixture(scope="module")
def indexed_merged(tmp_path_factory, indexed_shards) -> Path:
    out: Path = tmp_path_factory.mktemp("indexed_merged")
    merge_shards_module(indexed_shards, out)
    return out


@pytest.fixture(scope="module")
def dataset(indexed_merged, prior) -> TrainingDataset:
    return TrainingDataset(
        indexed_merged / "theta.npy", indexed_merged / "x.npy", prior
    )


@pytest.fixture(scope="module")
def nuis_prior(simulator_injector):
    # theta_1, theta_10..theta_19 are 11 params — 30 - 11 = 19
    return create_prior(simulator_injector, nuisance_pars=["theta_1*"])


class TestTrainingDataset:
    def test_length_counts_rows_across_shards(self, dataset):
        assert len(dataset) == N_SHARDS * ROWS_PER_SHARD

    def test_merged_conftest_data_length(self, merged_data_dir, prior, test_consts):
        ds = TrainingDataset(
            merged_data_dir / "theta.npy", merged_data_dir / "x.npy", prior
        )
        assert len(ds) == test_consts.n_files * test_consts.n_simulations

    def test_getitem_returns_single_row(self, dataset, test_consts):
        # A row in the first shard and one in a later shard
        for idx in (0, ROWS_PER_SHARD + 1):
            theta, x = dataset[idx]
            assert theta.dtype == x.dtype == torch.float32
            torch.testing.assert_close(
                theta, torch.full((test_consts.theta_dim,), float(idx))
            )
            torch.testing.assert_close(x, torch.full((test_consts.x_dim,), float(idx)))

    def test_slice_returns_batch(self, dataset, test_consts):
        theta, x = dataset[10:20]
        assert theta.shape == (10, test_consts.theta_dim)
        assert x.shape == (10, test_consts.x_dim)
        torch.testing.assert_close(x[:, 0], torch.arange(10, 20, dtype=torch.float32))

    def test_getitems_preserves_requested_order(self, dataset):
        # Unsorted, spans shards, and has gaps -- exercises the sort/coalesce
        # read path and the gather back into caller order.
        indices = [120, 3, 77, 4, 149, 0]
        theta, x = dataset.__getitems__(indices)
        expected = torch.tensor(indices, dtype=torch.float32)
        torch.testing.assert_close(theta[:, 0], expected)
        torch.testing.assert_close(x[:, 0], expected)

    def test_nuisance_filter_applied_on_read(
        self, indexed_merged, nuis_prior, test_consts
    ):
        ds = TrainingDataset(
            indexed_merged / "theta.npy", indexed_merged / "x.npy", nuis_prior
        )
        theta, _ = ds[0]
        assert len(theta) == test_consts.theta_dim - 11

    def test_prefiltered_merge_is_not_filtered_again(
        self, tmp_path, indexed_shards, nuis_prior, test_consts
    ):
        prior_path = tmp_path / "nuis_prior.pkl"
        nuis_prior.save(prior_path)
        merge_shards_module(indexed_shards, tmp_path / "merged", prior_path=prior_path)

        ds = TrainingDataset(
            tmp_path / "merged" / "theta.npy", tmp_path / "merged" / "x.npy", nuis_prior
        )
        theta, _ = ds[0]
        assert len(theta) == test_consts.theta_dim - 11

    def test_prefiltered_merge_rejects_different_prior(
        self, tmp_path, indexed_shards, nuis_prior, prior
    ):
        prior_path = tmp_path / "nuis_prior.pkl"
        nuis_prior.save(prior_path)
        merge_shards_module(indexed_shards, tmp_path / "merged", prior_path=prior_path)

        with pytest.raises(ValueError, match="different prior"):
            TrainingDataset(
                tmp_path / "merged" / "theta.npy", tmp_path / "merged" / "x.npy", prior
            )

    def test_mismatched_row_counts_raise(self, tmp_path, indexed_merged, prior):
        short_x = tmp_path / "x.npy"
        np.save(short_x, np.load(indexed_merged / "x.npy")[:-1])
        with pytest.raises(ValueError, match="do not match"):
            TrainingDataset(indexed_merged / "theta.npy", short_x, prior)

    @pytest.mark.parametrize(
        ("have_preadv", "have_pread"),
        [(True, True), (False, True), (False, False)],
        ids=["preadv", "pread", "seek_read"],
    )
    def test_read_paths_agree(
        self, monkeypatch, indexed_merged, prior, have_preadv, have_pread
    ):
        """Every platform read path returns the same rows (seek_read is Windows)."""
        if have_preadv and not hasattr(os, "preadv"):
            pytest.skip("os.preadv unavailable on this platform")
        if have_pread and not hasattr(os, "pread"):
            pytest.skip("os.pread unavailable on this platform")

        monkeypatch.setattr(TrainingDataset, "_HAVE_PREADV", have_preadv)
        monkeypatch.setattr(TrainingDataset, "_HAVE_PREAD", have_pread)
        ds = TrainingDataset(
            indexed_merged / "theta.npy", indexed_merged / "x.npy", prior
        )

        indices = [120, 3, 77, 4, 149, 0]
        theta, x = ds.__getitems__(indices)
        expected = torch.tensor(indices, dtype=torch.float32)
        torch.testing.assert_close(theta[:, 0], expected)
        torch.testing.assert_close(x[:, 0], expected)
