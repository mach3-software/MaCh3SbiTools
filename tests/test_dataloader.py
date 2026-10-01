"""
Tests for mach3sbitools.data_loaders.ParaketDataset.
"""

import pytest
import torch

from mach3sbitools.data_loaders import TrainingDataset
from mach3sbitools.simulator import create_prior


@pytest.fixture(scope="session")
def paraket_dataset(dummy_data_dir, prior):
    return TrainingDataset(dummy_data_dir, prior)


class TestParaketDataset:
    def test_file_count_and_dataset_length(
        self, paraket_dataset, dummy_data_dir, test_consts
    ):
        """Files on disk, dataset length, and item shapes in one pass."""
        n_feather = len(list(dummy_data_dir.glob("*.feather")))
        assert n_feather == test_consts.n_files
        assert len(paraket_dataset) == test_consts.n_files * test_consts.n_simulations

    def test_getitem_returns_correct_tensors(self, paraket_dataset, test_consts):
        # Check a row in the first file and one in a later file
        for idx in (0, test_consts.n_simulations + 1):
            theta, x = paraket_dataset[idx]
            torch.testing.assert_close(
                x, torch.from_numpy(test_consts.x[0]).to(torch.float32)
            )
            torch.testing.assert_close(
                theta, torch.from_numpy(test_consts.theta[0]).to(torch.float32)
            )

    def test_getitem_out_of_range(self, paraket_dataset):
        with pytest.raises(IndexError):
            paraket_dataset[len(paraket_dataset)]

    def test_nuisance_filter_reduces_theta_dim(
        self, dummy_data_dir, simulator_injector, test_consts
    ):

        nuis_prior = create_prior(simulator_injector, nuisance_pars=["theta_1*"])
        filtered = TrainingDataset(dummy_data_dir, nuis_prior)

        theta, _ = filtered[0]
        # theta_1, theta_10..theta_19 are 11 params — 30 - 11 = 19
        assert len(theta) == test_consts.theta_dim - 11

    def test_tensor_dataset_total_length(self, paraket_dataset, test_consts):
        ds = paraket_dataset.to_tensor_dataset()
        assert len(ds) == test_consts.n_files * test_consts.n_simulations
