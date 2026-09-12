"""
Tests for mach3sbitools.data_loaders.TrainingDataset.
"""

import numpy as np
import pytest
import torch

from mach3sbitools.data_loaders import TrainingDataset
from mach3sbitools.simulator import create_prior


@pytest.fixture(scope="session")
def training_dataset(merged_data_dir, prior):
    return TrainingDataset(
        merged_data_dir / "theta.npy", merged_data_dir / "x.npy", prior
    )


def _write_npy_pair(tmp_path, theta, x):
    """
    Write a ``(theta, x)`` pair as the memmap files the dataset expects.

    :param tmp_path: Directory to write into.
    :param theta: Parameter array.
    :param x: Observable array.
    :returns: Tuple of ``(theta_path, x_path)``.
    """
    theta_path, x_path = tmp_path / "theta.npy", tmp_path / "x.npy"
    np.save(theta_path, theta)
    np.save(x_path, x)
    return theta_path, x_path


class _FakePrior:
    """Minimal stand-in exposing only the nuisance mask the dataset reads."""

    def __init__(self, keep_mask: list[bool]):
        self.nuisance_filter = torch.tensor(keep_mask)


class TestTrainingDataset:
    def test_length_matches_merged_rows(self, training_dataset, test_consts):
        assert len(training_dataset) == (
            test_consts.n_files * test_consts.n_simulations
        )

    def test_dims_report_active_parameters(self, training_dataset, test_consts):
        assert training_dataset.theta_dim == test_consts.theta_dim
        assert training_dataset.x_dim == test_consts.x_dim

    def test_getitem_returns_float32_pair(self, training_dataset, test_consts):
        theta, x = training_dataset[0]
        assert theta.shape == (test_consts.theta_dim,)
        assert x.shape == (test_consts.x_dim,)
        assert theta.dtype == torch.float32 and x.dtype == torch.float32

    def test_slice_returns_batch(self, training_dataset, test_consts):
        theta, x = training_dataset[:8]
        assert theta.shape == (8, test_consts.theta_dim)
        assert x.shape == (8, test_consts.x_dim)

    def test_getitems_returns_a_stacked_batch(self, training_dataset, test_consts):
        """
        The batch comes back already stacked.

        Returning a list of rows would have the collate function restack into
        exactly this tensor, which for a real batch costs more than the read.
        """
        theta, x = training_dataset.__getitems__([0, 3, 7])
        assert theta.shape == (3, test_consts.theta_dim)
        assert x.shape == (3, test_consts.x_dim)

    def test_getitems_rows_match_getitem(self, training_dataset):
        indices = [0, 3, 7]
        theta, x = training_dataset.__getitems__(indices)
        for position, index in enumerate(indices):
            row_theta, row_x = training_dataset[index]
            torch.testing.assert_close(theta[position], row_theta)
            torch.testing.assert_close(x[position], row_x)

    def test_nuisance_filter_reduces_theta_dim(
        self, merged_data_dir, simulator_injector, test_consts
    ):
        nuis_prior = create_prior(simulator_injector, nuisance_pars=["theta_1*"])
        filtered = TrainingDataset(
            merged_data_dir / "theta.npy", merged_data_dir / "x.npy", nuis_prior
        )
        # theta_1, theta_10..theta_19 are 11 params
        expected = test_consts.theta_dim - 11
        assert filtered.theta_dim == expected
        assert filtered[0][0].shape == (expected,)


class TestTrainingDatasetAxisHandling:
    """
    The nuisance filter must always apply to the parameter axis.

    A shape heuristic that compares ``theta.shape[0]`` against the filter
    length silently transposes the result whenever a batch happens to be as
    long as the parameter count, so square batches are the regression case.
    """

    @pytest.mark.parametrize("n_rows", [4, 9])
    def test_filter_applies_to_last_axis(self, tmp_path, n_rows):
        n_params = 4
        theta = np.arange(n_rows * n_params, dtype=np.float32).reshape(n_rows, n_params)
        x = np.zeros((n_rows, 3), dtype=np.float32)
        theta_path, x_path = _write_npy_pair(tmp_path, theta, x)

        keep = [True, False, True, False]
        dataset = TrainingDataset(theta_path, x_path, _FakePrior(keep))

        batch, _ = dataset[:n_rows]
        assert batch.shape == (n_rows, 2)
        np.testing.assert_allclose(batch.numpy(), theta[:, keep])

    def test_single_row_filtered_on_parameter_axis(self, tmp_path):
        theta = np.arange(12, dtype=np.float32).reshape(3, 4)
        x = np.zeros((3, 3), dtype=np.float32)
        theta_path, x_path = _write_npy_pair(tmp_path, theta, x)

        keep = [True, False, True, False]
        dataset = TrainingDataset(theta_path, x_path, _FakePrior(keep))

        row, _ = dataset[1]
        np.testing.assert_allclose(row.numpy(), theta[1, keep])


class TestTrainingDatasetValidation:
    def test_rejects_mismatched_row_counts(self, tmp_path):
        theta_path, x_path = _write_npy_pair(
            tmp_path, np.zeros((10, 4), np.float32), np.zeros((8, 3), np.float32)
        )
        with pytest.raises(ValueError, match="row count"):
            TrainingDataset(theta_path, x_path, _FakePrior([True] * 4))

    def test_rejects_theta_width_prior_mismatch(self, tmp_path):
        theta_path, x_path = _write_npy_pair(
            tmp_path, np.zeros((10, 4), np.float32), np.zeros((10, 3), np.float32)
        )
        with pytest.raises(ValueError, match="parameters"):
            TrainingDataset(theta_path, x_path, _FakePrior([True] * 6))
