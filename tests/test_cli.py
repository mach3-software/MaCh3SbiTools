"""
Wiring tests for the mach3sbi CLI.

These check that each subcommand parses its options and calls into the right
application module with them. The behaviour behind those modules is tested
directly elsewhere, so the heavy objects are mocked out here.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mach3sbitools.apps.main_cli import cli


@pytest.fixture()
def tmp_files(tmp_path):
    """Paths the CLI's `exists=True` options need to resolve."""
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

    # Must have a "data" column — matches pq.read_table(...)["data"] in the
    # inference command.
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


@pytest.fixture()
def simulator_args(tmp_files):
    """The -m/-s/-c trio every simulator-backed command requires."""
    return [
        "-m",
        "mypackage.simulator",
        "-s",
        "MySimulator",
        "-c",
        str(tmp_files["config"]),
    ]


@pytest.fixture()
def train_args(tmp_files):
    """The -r/-d/-s trio the train command requires."""
    return [
        "-r",
        str(tmp_files["prior"]),
        "-d",
        str(tmp_files["data_dir"]),
        "-s",
        str(tmp_files["tmp"] / "models"),
    ]


@pytest.fixture()
def inference_args(tmp_files):
    """The options the inference command requires."""
    return [
        "-i",
        str(tmp_files["checkpoint"]),
        "-r",
        str(tmp_files["prior"]),
        "-s",
        str(tmp_files["inference"] / "samples.parquet"),
        "-o",
        str(tmp_files["observed"]),
        "-n",
        "100",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Group
# ─────────────────────────────────────────────────────────────────────────────


def test_help(runner):
    assert runner.invoke(cli, ["--help"]).exit_code == 0


def test_unknown_command(runner):
    assert runner.invoke(cli, ["not_a_command"]).exit_code != 0


@pytest.mark.parametrize(
    "command",
    ["create_prior", "simulate", "save_data", "train", "inference", "merge-shards"],
)
def test_every_command_has_help(runner, command):
    """A missing docstring or a broken decorator stack shows up here."""
    result = runner.invoke(cli, [command, "--help"])
    assert result.exit_code == 0, result.output


# ─────────────────────────────────────────────────────────────────────────────
# Commands reach their application module
# ─────────────────────────────────────────────────────────────────────────────


def test_create_prior_runs(runner, simulator_args, tmp_files):
    with (
        patch("mach3sbitools.apps.save_prior.create_prior") as create,
        patch("mach3sbitools.apps.save_prior.get_simulator") as get_sim,
    ):
        result = runner.invoke(
            cli,
            ["create_prior", *simulator_args, "-o", str(tmp_files["tmp"] / "p.pkl")],
        )

    assert result.exit_code == 0, result.output
    get_sim.assert_called_once()
    create.assert_called_once()


def test_simulate_runs(runner, simulator_args):
    with patch("mach3sbitools.apps.simulate.Simulator") as sim_cls:
        sim_cls.return_value.simulate.return_value = (MagicMock(), MagicMock())
        result = runner.invoke(
            cli, ["simulate", *simulator_args, "-o", "out.feather", "-n", "100"]
        )

    assert result.exit_code == 0, result.output
    sim_cls.return_value.simulate.assert_called_once()
    sim_cls.return_value.save.assert_called_once()


def test_save_data_runs(runner, simulator_args):
    with patch("mach3sbitools.apps.save_data.Simulator") as sim_cls:
        result = runner.invoke(cli, ["save_data", *simulator_args, "-o", "obs.parquet"])

    assert result.exit_code == 0, result.output
    sim_cls.return_value.save_data.assert_called_once()


def test_train_runs(runner, train_args):
    with patch("mach3sbitools.apps.train.InferenceHandler") as handler_cls:
        result = runner.invoke(cli, ["train", *train_args])

    assert result.exit_code == 0, result.output
    handler = handler_cls.return_value
    handler.set_dataset.assert_called_once()
    handler.create_posterior.assert_called_once()
    handler.train_posterior.assert_called_once()


@pytest.mark.slow
def test_inference_runs(runner, inference_args):
    with (
        patch("mach3sbitools.apps.inference.pairplot"),
        patch("mach3sbitools.apps.inference.InferenceHandler") as handler_cls,
    ):
        handler = handler_cls.return_value
        handler.prior.prior_data.parameter_names = np.array(["p1"])
        handler.sample_posterior.return_value.cpu.return_value.numpy.return_value = (
            np.random.randn(100, 1)
        )
        result = runner.invoke(cli, ["inference", *inference_args])

    assert result.exit_code == 0, result.output
    handler.load_posterior.assert_called_once()
    handler.sample_posterior.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Options reach the config objects
# ─────────────────────────────────────────────────────────────────────────────


class TestTrainOptionsForwarded:
    @pytest.fixture()
    def train_configs(self, runner, train_args):
        """Invoke train with extra flags and return the configs it built."""

        def _invoke(*extra):
            with patch("mach3sbitools.apps.train.InferenceHandler") as handler_cls:
                result = runner.invoke(cli, ["train", *train_args, *extra])
                assert result.exit_code == 0, result.output
                handler = handler_cls.return_value
                return (
                    handler.train_posterior.call_args[0][0],
                    handler.create_posterior.call_args[0][0],
                )

        return _invoke

    @pytest.mark.parametrize(
        "flag,value,field,expected",
        [
            ("--batch_size", "512", "batch_size", 512),
            ("--max_epochs", "10", "max_epochs", 10),
            ("--learning_rate", "0.01", "learning_rate", 0.01),
            ("--num_workers", "4", "num_workers", 4),
            ("--stop_after_epochs", "9", "stop_after_epochs", 9),
        ],
    )
    def test_training_options(self, train_configs, flag, value, field, expected):
        training_config, _ = train_configs(flag, value)
        assert getattr(training_config, field) == expected

    @pytest.mark.parametrize(
        "flag,field",
        [("--show_progress", "show_progress"), ("--compile_model", "compile")],
    )
    def test_training_flags(self, train_configs, flag, field):
        training_config, _ = train_configs(flag)
        assert getattr(training_config, field) is True

    @pytest.mark.parametrize(
        "flag,value,field,expected",
        [
            ("--model", "nsf", "model", "nsf"),
            ("--hidden", "64", "hidden_features", 64),
            ("--transforms", "3", "num_transforms", 3),
            ("--num_bins", "7", "num_bins", 7),
        ],
    )
    def test_architecture_options(self, train_configs, flag, value, field, expected):
        _, posterior_config = train_configs(flag, value)
        assert getattr(posterior_config, field) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Required options are enforced
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["create_prior", "-m", "pkg.sim"], id="create_prior/no-class"),
        pytest.param(["simulate", "-m", "pkg.sim", "-o", "x"], id="simulate/no-n"),
        pytest.param(["train", "-d", "sims/", "-s", "models/"], id="train/no-prior"),
        pytest.param(["train", "-r", "p.pkl", "-s", "models/"], id="train/no-dataset"),
        pytest.param(["train", "-r", "p.pkl", "-d", "sims/"], id="train/no-save"),
        pytest.param(["inference", "-n", "100"], id="inference/no-posterior"),
    ],
)
def test_missing_required_options_fail(runner, args):
    assert runner.invoke(cli, args).exit_code != 0


def test_nonexistent_paths_are_rejected(runner, tmp_files):
    """`exists=True` options must reject a path that isn't there."""
    result = runner.invoke(
        cli,
        [
            "inference",
            "-i",
            "/no/such/checkpoint.pt",
            "-r",
            str(tmp_files["prior"]),
            "-s",
            "out.parquet",
            "-o",
            str(tmp_files["observed"]),
            "-n",
            "100",
        ],
    )
    assert result.exit_code != 0
