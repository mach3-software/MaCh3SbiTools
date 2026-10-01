"""
Lightweight smoke tests for the mach3sbi CLI.

Verifies each subcommand is wired up and required arguments are enforced.
Business logic is tested elsewhere.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from mach3sbitools.apps.main_cli import cli


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture()
def tmp_files(tmp_path):
    config = tmp_path / "config.yaml"
    prior = tmp_path / "prior.pkl"
    checkpoint = tmp_path / "best.pt"
    observed = tmp_path / "observed.parquet"
    data_dir = tmp_path / "sims"
    inference = tmp_path / "inference"

    config.touch()
    prior.touch()
    checkpoint.touch()
    data_dir.mkdir()
    inference.mkdir()

    # Must have a "data" column — matches pq.read_table(...)["data"] in inference cmd
    pd.DataFrame({"data": np.random.poisson(10, size=50).astype(float)}).to_parquet(
        observed
    )

    return {
        "config": config,
        "prior": prior,
        "checkpoint": checkpoint,
        "observed": observed,
        "data_dir": data_dir,
        "tmp": tmp_path,
        "inference": inference,
    }


# ---------------------------------------------------------------------------
# Group
# ---------------------------------------------------------------------------


def test_help(runner):
    assert runner.invoke(cli, ["--help"]).exit_code == 0


def test_unknown_command(runner):
    assert runner.invoke(cli, ["not_a_command"]).exit_code != 0


# ---------------------------------------------------------------------------
# create_prior
# ---------------------------------------------------------------------------


@patch("mach3sbitools.apps.save_prior.create_prior", return_value=MagicMock())
@patch("mach3sbitools.apps.save_prior.get_simulator", return_value=MagicMock())
def test_create_prior_runs(mock_sim, mock_prior, runner, tmp_files):
    result = runner.invoke(
        cli,
        [
            "create_prior",
            "-m",
            "mypackage.simulator",
            "-s",
            "MySimulator",
            "-c",
            str(tmp_files["config"]),
            "-o",
            str(tmp_files["tmp"] / "prior.pkl"),
        ],
    )
    assert result.exit_code == 0, result.output
    mock_sim.assert_called_once()
    mock_prior.assert_called_once()


def test_create_prior_missing_args(runner, tmp_files):
    result = runner.invoke(
        cli,
        ["create_prior", "-m", "mypackage.simulator", "-c", str(tmp_files["config"])],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------


@patch("mach3sbitools.apps.simulate.Simulator")
def test_simulate_runs(mock_sim_cls, runner, tmp_files):
    mock_sim = MagicMock()
    mock_sim.simulate.return_value = (MagicMock(), MagicMock())
    mock_sim_cls.return_value = mock_sim

    result = runner.invoke(
        cli,
        [
            "simulate",
            "-m",
            "mypackage.simulator",
            "-s",
            "MySimulator",
            "-c",
            str(tmp_files["config"]),
            "-o",
            "out.feather",
            "-n",
            "100",
        ],
    )
    assert result.exit_code == 0, result.output
    mock_sim.simulate.assert_called_once()
    mock_sim.save.assert_called_once()


def test_simulate_missing_n(runner, tmp_files):
    result = runner.invoke(
        cli,
        [
            "simulate",
            "-m",
            "mypackage.simulator",
            "-s",
            "MySimulator",
            "-c",
            str(tmp_files["config"]),
            "-o",
            "out.feather",
        ],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# save_data
# ---------------------------------------------------------------------------


@patch("mach3sbitools.apps.save_data.Simulator")
def test_save_data_runs(mock_sim_cls, runner, tmp_files):
    mock_sim_cls.return_value = MagicMock()
    result = runner.invoke(
        cli,
        [
            "save_data",
            "-m",
            "mypackage.simulator",
            "-s",
            "MySimulator",
            "-c",
            str(tmp_files["config"]),
            "-o",
            "obs.parquet",
        ],
    )
    assert result.exit_code == 0, result.output
    mock_sim_cls.return_value.save_data.assert_called_once()


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@patch("mach3sbitools.apps.train.InferenceHandler")
def test_train_runs(mock_handler_cls, runner, tmp_files):
    mock_handler_cls.return_value = MagicMock()
    result = runner.invoke(
        cli,
        [
            "train",
            "-r",
            str(tmp_files["prior"]),
            "-d",
            str(tmp_files["data_dir"]),
            "-s",
            str(tmp_files["tmp"] / "models"),
        ],
    )
    assert result.exit_code == 0, result.output
    h = mock_handler_cls.return_value
    h.set_dataset.assert_called_once()
    h.create_posterior.assert_called_once()
    h.train_posterior.assert_called_once()


@pytest.mark.parametrize(
    "args",
    [
        ["train", "-d", "sims/", "-s", "models/"],  # missing -r
        ["train", "-r", "prior.pkl", "-s", "models/"],  # missing -d
        ["train", "-r", "prior.pkl", "-d", "sims/"],  # missing -s
    ],
)
def test_train_missing_required(runner, args):
    assert runner.invoke(cli, args).exit_code != 0


@patch("mach3sbitools.apps.train.InferenceHandler")
def test_train_config_forwarded(mock_handler_cls, runner, tmp_files):
    """Key training options reach TrainingConfig."""
    mock_handler_cls.return_value = MagicMock()
    runner.invoke(
        cli,
        [
            "train",
            "-r",
            str(tmp_files["prior"]),
            "-d",
            str(tmp_files["data_dir"]),
            "-s",
            str(tmp_files["tmp"] / "models"),
            "--batch_size",
            "512",
            "--max_epochs",
            "10",
            "--show_progress",
            "--compile_model",
        ],
    )
    cfg = mock_handler_cls.return_value.train_posterior.call_args[0][0]
    assert cfg.batch_size == 512
    assert cfg.max_epochs == 10
    assert cfg.show_progress is True
    assert cfg.compile is True


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------


@pytest.mark.slow
@patch("mach3sbitools.apps.inference.pairplot")
@patch("mach3sbitools.apps.inference.InferenceHandler")
def test_inference_runs(mock_handler_cls, mock_pairplot, runner, tmp_files):
    mock_handler = MagicMock()
    mock_handler.prior.prior_data.parameter_names = np.array(["p1"])

    # Shape must be (n_samples, len(parameter_names)) — here (100, 1)
    fake_samples = np.random.randn(100, 1)
    mock_handler.sample_posterior.return_value.cpu.return_value.numpy.return_value = (
        fake_samples
    )

    mock_handler_cls.return_value = mock_handler

    result = runner.invoke(
        cli,
        [
            "inference",
            "-i",
            str(tmp_files["checkpoint"]),
            "-r",
            str(tmp_files["prior"]),
            "-s",
            str(tmp_files["inference"] / "samples.parquet"),
            "-n",
            "100",
            "-o",
            str(tmp_files["observed"]),
        ],
    )
    assert result.exit_code == 0, result.output
    mock_handler.load_posterior.assert_called_once()
    mock_handler.sample_posterior.assert_called_once()


def test_inference_missing_posterior(runner, tmp_files):
    result = runner.invoke(
        cli,
        [
            "inference",
            "--posterior",
            "/nope.pt",
            "-r",
            str(tmp_files["prior"]),
            "-n",
            "100",
            "-o",
            str(tmp_files["observed"]),
        ],
    )
    assert result.exit_code != 0


def test_inference_missing_prior(runner, tmp_files):
    result = runner.invoke(
        cli,
        [
            "inference",
            "-i",
            str(tmp_files["checkpoint"]),
            "-n",
            "100",
            "-o",
            str(tmp_files["observed"]),
        ],
    )
    assert result.exit_code != 0
