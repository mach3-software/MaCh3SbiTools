"""
Tests for mach3sbitools.data_loaders.ParaketDataset.
"""

import pytest
import torch

from mach3sbitools.data_loaders import TrainingDataset
from mach3sbitools.simulator import create_prior
from mach3sbitools.utils import TorchDeviceHandler

device_handler = TorchDeviceHandler()


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
        assert len(paraket_dataset) == test_consts.n_files

    def test_getitem_returns_correct_tensors(self, paraket_dataset, test_consts):
        theta, x = paraket_dataset[0]
        torch.testing.assert_close(
            device_handler.to_tensor(x), device_handler.to_tensor(test_consts.x)
        )
        torch.testing.assert_close(
            device_handler.to_tensor(theta), device_handler.to_tensor(test_consts.theta)
        )

    def test_nuisance_filter_reduces_theta_dim(
        self, dummy_data_dir, simulator_injector, test_consts
    ):

        nuis_prior = create_prior(simulator_injector, nuisance_pars=["theta_1*"])
        filtered = TrainingDataset(dummy_data_dir, nuis_prior)

        theta, _ = filtered[0]
        # theta_1, theta_10..theta_19 are 11 params — 30 - 11 = 19
        assert len(theta[0]) == test_consts.theta_dim - 11

    def test_tensor_dataset_total_length(self, paraket_dataset, test_consts):
        ds = paraket_dataset.to_tensor_dataset()
        assert len(ds) == test_consts.n_files * test_consts.n_simulations

    def test_random_subsample_shapes_and_cap(self, paraket_dataset, test_consts):
        n_total = len(paraket_dataset)

        theta, x = paraket_dataset.random_subsample(5)
        assert theta.shape[0] == 5
        assert x.shape[0] == 5
        assert theta.shape[1] == test_consts.theta_dim

        # Requesting more rows than exist should cap at the dataset length,
        # not raise or loop forever.
        theta_all, x_all = paraket_dataset.random_subsample(n_total + 1000)
        assert theta_all.shape[0] == n_total
        assert x_all.shape[0] == n_total

    def test_random_subsample_is_reproducible_with_generator(self, paraket_dataset):
        gen1 = torch.Generator().manual_seed(123)
        gen2 = torch.Generator().manual_seed(123)
        theta_a, _ = paraket_dataset.random_subsample(5, generator=gen1)
        theta_b, _ = paraket_dataset.random_subsample(5, generator=gen2)
        torch.testing.assert_close(theta_a, theta_b)
