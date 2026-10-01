from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from mach3sbitools.apps.merge_shards import merge_shards_module
from mach3sbitools.simulator import create_prior
from mach3sbitools.simulator.simulator_injector import get_simulator
from mach3sbitools.types import SimulatorData
from mach3sbitools.utils import MaCh3Logger, PosteriorConfig, TrainingConfig, to_feather

"""
Generic test configuration
"""

MaCh3Logger(level="INFO")


# Consts.
@dataclass(frozen=True)
class TestConsts:
    n_files: int = 10
    theta_dim: int = 30
    x_dim: int = 12
    n_simulations: int = 1000

    @property
    def x(self) -> SimulatorData:
        return np.ones((self.n_simulations, self.x_dim), dtype=np.float64)

    @property
    def theta(self) -> SimulatorData:
        return np.ones((self.n_simulations, self.theta_dim), dtype=np.float64)

    @property
    def parameter_names(self) -> list[str]:
        return [f"theta_{i}" for i in range(1, self.theta_dim + 1)]


@pytest.fixture(scope="session")
def test_consts() -> TestConsts:
    return TestConsts()


@pytest.fixture(scope="session")
def simulator_module() -> str:
    return "dummy_simulator"


@pytest.fixture(scope="session")
def simulator_class() -> str:
    return "DummySimulator"


@pytest.fixture(scope="session")
def dummy_config(tmp_path_factory) -> Path:
    # Dummy configuration
    dummy_config_dir: Path = tmp_path_factory.mktemp("dummy_configs")
    dummy_config_file = dummy_config_dir / "dummy_config.yaml"
    dummy_config_file.touch()

    return dummy_config_file


def generate_data(data_folder: Path, test_consts: TestConsts) -> None:
    for i in range(test_consts.n_files):
        file = data_folder / f"tmp_data{i}.feather"
        to_feather(file, test_consts.theta, test_consts.x)


@pytest.fixture(scope="session")
def dummy_data_dir(tmp_path_factory, test_consts) -> Path:
    data_folder: Path = tmp_path_factory.mktemp("data")
    generate_data(data_folder, test_consts)
    return data_folder


@pytest.fixture(scope="session")
def merged_data_dir(tmp_path_factory, dummy_data_dir) -> Path:
    """The feather shards in ``dummy_data_dir`` merged into theta.npy / x.npy."""
    merged_folder: Path = tmp_path_factory.mktemp("merged")
    merge_shards_module(dummy_data_dir, merged_folder)
    return merged_folder


@pytest.fixture(scope="session")
def simulator_injector(simulator_module, simulator_class, dummy_config):
    return get_simulator(simulator_module, simulator_class, dummy_config)


# First let's check a flat/non-flat prior
@pytest.fixture(scope="session")
def prior(simulator_injector):
    # Need the injector
    return create_prior(simulator_injector, cyclical_pars=["theta_9"])


@pytest.fixture(scope="session")
def prior_save(prior, tmp_path_factory) -> Path:
    prior_dir: Path = tmp_path_factory.mktemp("priors")
    prior_path = prior_dir / "prior.pkl"
    prior.save(prior_path)
    return prior_path


@pytest.fixture(scope="session")
def posterior_config():
    return PosteriorConfig(
        model="maf",
        hidden_features=5,
        num_transforms=2,
        dropout_probability=0,
        num_blocks=2,
        num_bins=10,
    )


@pytest.fixture(scope="session")
def model_save_path(tmp_path_factory) -> Path:
    return cast(Path, tmp_path_factory.mktemp("models") / "test_model.ckpt")


@pytest.fixture(scope="session")
def training_config(model_save_path):
    return TrainingConfig(
        save_path=model_save_path,
        learning_rate=0.001,
        batch_size=256,
        max_epochs=10,
        autosave_every=500,
        print_interval=100,
        show_progress=True,
    )
