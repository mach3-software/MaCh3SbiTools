"""
Tests for mach3sbitools.data_loaders.StreamingFeatherDataset.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from mach3sbitools.data_loaders import CompressedDatasetWrapper, StreamingFeatherDataset
from mach3sbitools.simulator import create_prior
from mach3sbitools.utils import TorchDeviceHandler, to_feather

device_handler = TorchDeviceHandler()


@pytest.fixture
def varied_data_dir(tmp_path_factory, test_consts) -> Path:
    """Several shards of *distinct* random rows (unlike dummy_data_dir's
    constant-valued shards) so per-row correctness is actually exercised.
    Uses the same theta/x dimensions as test_consts so it's compatible
    with the shared `prior`/`simulator_injector` fixtures."""
    data_folder: Path = tmp_path_factory.mktemp("varied_data")
    rng = np.random.default_rng(0)
    for i in range(5):
        n = int(rng.integers(20, 60))
        theta = rng.random((n, test_consts.theta_dim)).astype(np.float64)
        x = rng.random((n, test_consts.x_dim)).astype(np.float64)
        to_feather(data_folder / f"shard_{i}.feather", theta, x)
    return data_folder


def _concat_ground_truth(data_folder: Path) -> tuple[np.ndarray, np.ndarray]:
    from mach3sbitools.utils import from_feather

    thetas, xs = [], []
    for f in sorted(data_folder.glob("*.feather")):
        th, x = from_feather(f)
        thetas.append(th)
        xs.append(x)
    return np.concatenate(thetas), np.concatenate(xs)


class TestStreamingFeatherDataset:
    def test_length_matches_total_rows(self, varied_data_dir, prior):
        ds = StreamingFeatherDataset(varied_data_dir, prior, cache_size=2)
        theta_gt, _ = _concat_ground_truth(varied_data_dir)
        assert len(ds) == theta_gt.shape[0]

    def test_missing_folder_raises(self, tmp_path, prior):
        with pytest.raises(FileNotFoundError):
            StreamingFeatherDataset(tmp_path / "does_not_exist", prior)

    def test_getitem_matches_ground_truth_for_every_row(self, varied_data_dir, prior):
        ds = StreamingFeatherDataset(varied_data_dir, prior, cache_size=1)
        theta_gt, x_gt = _concat_ground_truth(varied_data_dir)

        for i in range(len(ds)):
            theta, x = ds[i]
            torch.testing.assert_close(
                device_handler.to_tensor(theta.numpy()),
                device_handler.to_tensor(theta_gt[i]),
            )
            torch.testing.assert_close(
                device_handler.to_tensor(x.numpy()), device_handler.to_tensor(x_gt[i])
            )

    def test_negative_index(self, varied_data_dir, prior):
        ds = StreamingFeatherDataset(varied_data_dir, prior, cache_size=2)
        theta_last, x_last = ds[-1]
        theta_explicit, x_explicit = ds[len(ds) - 1]
        torch.testing.assert_close(theta_last, theta_explicit)
        torch.testing.assert_close(x_last, x_explicit)

    def test_out_of_range_index_raises(self, varied_data_dir, prior):
        ds = StreamingFeatherDataset(varied_data_dir, prior, cache_size=2)
        with pytest.raises(IndexError):
            _ = ds[len(ds)]

    def test_dataloader_with_shuffle_and_workers_sees_every_row_once(
        self, varied_data_dir, prior
    ):
        ds = StreamingFeatherDataset(varied_data_dir, prior, cache_size=2)
        dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2)

        seen = 0
        for theta_batch, _ in dl:
            seen += theta_batch.shape[0]
        assert seen == len(ds)

    def test_cache_size_one_still_correct(self, varied_data_dir, prior):
        """The minimum-memory (cache_size=1) config must still read the
        right row even when iteration order forces constant re-loading."""
        ds = StreamingFeatherDataset(varied_data_dir, prior, cache_size=1)
        theta_gt, _ = _concat_ground_truth(varied_data_dir)
        # touch rows out of shard order to force cache thrash
        for i in reversed(range(0, len(ds), 7)):
            theta, _ = ds[i]
            torch.testing.assert_close(
                device_handler.to_tensor(theta.numpy()),
                device_handler.to_tensor(theta_gt[i]),
            )

    def test_sample_rows_shape_and_membership(self, varied_data_dir, prior):
        ds = StreamingFeatherDataset(varied_data_dir, prior, cache_size=2)
        theta_gt, _ = _concat_ground_truth(varied_data_dir)

        n = 17
        theta_s, x_s = ds.sample_rows(n, seed=3)
        assert theta_s.shape[0] == n
        assert x_s.shape[0] == n

        # every sampled theta row should exist somewhere in the ground truth
        for row in theta_s.numpy():
            assert np.any(np.all(np.isclose(theta_gt, row), axis=1))

    def test_sample_rows_capped_at_dataset_size(self, varied_data_dir, prior):
        ds = StreamingFeatherDataset(varied_data_dir, prior, cache_size=2)
        theta_s, _ = ds.sample_rows(n=10**9, seed=0)
        assert theta_s.shape[0] == len(ds)

    def test_manifest_is_cached_and_reused(self, varied_data_dir, prior):
        ds1 = StreamingFeatherDataset(varied_data_dir, prior, cache_size=2)
        manifest = varied_data_dir / ".mach3sbi_row_index.json"
        assert manifest.exists()

        ds2 = StreamingFeatherDataset(varied_data_dir, prior, cache_size=2)
        assert ds1.row_counts == ds2.row_counts

    def test_rebuild_manifest_forces_recompute(self, varied_data_dir, prior):
        ds1 = StreamingFeatherDataset(varied_data_dir, prior, cache_size=2)
        ds2 = StreamingFeatherDataset(
            varied_data_dir, prior, cache_size=2, rebuild_manifest=True
        )
        assert ds1.row_counts == ds2.row_counts

    def test_nuisance_filter_reduces_theta_dim(
        self, varied_data_dir, simulator_injector, test_consts
    ):
        nuis_prior = create_prior(simulator_injector, nuisance_pars=["theta_1*"])
        ds = StreamingFeatherDataset(varied_data_dir, nuis_prior, cache_size=2)
        theta, _ = ds[0]
        # theta_1, theta_10..theta_19 are 11 params — 30 - 11 = 19
        assert theta.shape[0] == test_consts.theta_dim - 11

    def test_no_feather_files_raises(self, tmp_path, prior):
        tmp_path.mkdir(exist_ok=True)
        with pytest.raises(FileNotFoundError):
            StreamingFeatherDataset(tmp_path, prior)


class TestCompressedDatasetWrapper:
    class _Identity:
        def transform(self, arr):
            return arr * 2.0

    def test_wraps_and_transforms_each_item(self, varied_data_dir, prior):
        ds = StreamingFeatherDataset(varied_data_dir, prior, cache_size=2)
        wrapped = CompressedDatasetWrapper(
            ds, theta_compressor=self._Identity(), x_compressor=self._Identity()
        )

        assert len(wrapped) == len(ds)
        theta_raw, x_raw = ds[0]
        theta_wrapped, x_wrapped = wrapped[0]
        torch.testing.assert_close(theta_wrapped, theta_raw * 2.0)
        torch.testing.assert_close(x_wrapped, x_raw * 2.0)

    def test_no_compressors_is_passthrough(self, varied_data_dir, prior):
        ds = StreamingFeatherDataset(varied_data_dir, prior, cache_size=2)
        wrapped = CompressedDatasetWrapper(ds)
        theta_raw, x_raw = ds[0]
        theta_wrapped, x_wrapped = wrapped[0]
        torch.testing.assert_close(theta_wrapped, theta_raw)
        torch.testing.assert_close(x_wrapped, x_raw)
