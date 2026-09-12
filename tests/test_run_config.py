"""
Tests for the YAML run configuration and its interaction with the CLI.

The contract is: the file supplies defaults, an explicit flag always wins,
and anything the file gets wrong fails loudly rather than being ignored.
"""

from unittest.mock import MagicMock, patch

import pytest

from mach3sbitools.apps.main_cli import cli
from mach3sbitools.utils import RunConfigError, load_run_config


@pytest.fixture()
def write_config(tmp_path):
    """Write *text* to a YAML file and return its path."""

    def _write(text: str, name: str = "run.yaml"):
        path = tmp_path / name
        path.write_text(text)
        return path

    return _write


@pytest.fixture()
def train_paths(tmp_path):
    prior = tmp_path / "prior.pkl"
    prior.touch()
    dataset = tmp_path / "sims"
    dataset.mkdir()
    return {"prior": prior, "dataset": dataset, "save": tmp_path / "models" / "m.ckpt"}


# ─────────────────────────────────────────────────────────────────────────────
# load_run_config
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadRunConfig:
    def test_command_section_becomes_a_default_map_entry(self, write_config):
        path = write_config("train:\n  hidden: 512\n  max_epochs: 4321\n")
        assert load_run_config(path, cli)["train"] == {
            "hidden": 512,
            "max_epochs": 4321,
        }

    def test_logging_section_lands_at_the_top_level(self, write_config):
        path = write_config("logging:\n  log_level: WARNING\n")
        assert load_run_config(path, cli)["log_level"] == "WARNING"

    def test_shared_section_reaches_every_command_that_accepts_it(self, write_config):
        path = write_config("simulator:\n  simulator_class: MySim\n")
        default_map = load_run_config(path, cli)
        for command in ("create_prior", "simulate", "save_data", "diagnostics"):
            assert default_map[command]["simulator_class"] == "MySim"

    def test_shared_keys_skip_commands_that_do_not_accept_them(self, write_config):
        path = write_config("simulator:\n  simulator_class: MySim\n")
        default_map = load_run_config(path, cli)
        assert "simulator_class" not in default_map.get("train", {})

    def test_command_section_wins_over_the_shared_section(self, write_config):
        path = write_config(
            "simulator:\n  simulator_class: Shared\n"
            "simulate:\n  simulator_class: Specific\n"
        )
        default_map = load_run_config(path, cli)
        assert default_map["simulate"]["simulator_class"] == "Specific"
        assert default_map["save_data"]["simulator_class"] == "Shared"

    def test_hyphenated_command_names_are_accepted(self, write_config):
        path = write_config("merge_shards:\n  simulation_dir: shards/\n")
        default_map = load_run_config(path, cli)
        assert default_map["merge-shards"]["simulation_dir"] == "shards/"

    def test_empty_file_is_an_empty_map(self, write_config):
        assert load_run_config(write_config(""), cli) == {}


class TestRunConfigErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(RunConfigError, match="not found"):
            load_run_config(tmp_path / "nope.yaml", cli)

    def test_unknown_section_lists_the_real_commands(self, write_config):
        path = write_config("trian:\n  hidden: 512\n")
        with pytest.raises(RunConfigError) as exc:
            load_run_config(path, cli)
        assert "not a mach3sbi command" in str(exc.value)
        assert "train" in str(exc.value)

    def test_unknown_option_lists_the_valid_ones(self, write_config):
        path = write_config("train:\n  hiden: 512\n")
        with pytest.raises(RunConfigError) as exc:
            load_run_config(path, cli)
        assert "hiden" in str(exc.value)
        assert "hidden" in str(exc.value)

    def test_shared_option_no_command_accepts(self, write_config):
        path = write_config("simulator:\n  not_an_option: 1\n")
        with pytest.raises(RunConfigError, match="no command accepts"):
            load_run_config(path, cli)

    def test_section_must_be_a_mapping(self, write_config):
        with pytest.raises(RunConfigError, match="must be a mapping"):
            load_run_config(write_config("train: 5\n"), cli)

    def test_file_must_be_a_mapping(self, write_config):
        with pytest.raises(RunConfigError, match="mapping of sections"):
            load_run_config(write_config("- just\n- a\n- list\n"), cli)

    def test_group_title_placeholders_are_not_settable(self, write_config):
        """click-option-group's internal params must not leak into the schema."""
        with pytest.raises(RunConfigError) as exc:
            load_run_config(write_config("train:\n  nope: 1\n"), cli)
        assert "fake_" not in str(exc.value)

    def test_placeholder_names_are_rejected_like_any_typo(self, write_config):
        placeholder = next(
            p.name for p in cli.commands["train"].params if not p.expose_value
        )
        with pytest.raises(RunConfigError, match="unknown option"):
            load_run_config(write_config(f"train:\n  {placeholder}: 1\n"), cli)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end through the CLI
# ─────────────────────────────────────────────────────────────────────────────


class TestRunConfigThroughCli:
    def _train_config(self, write_config, train_paths, **extra):
        body = "\n".join(f"  {k}: {v}" for k, v in extra.items())
        return write_config(
            "train:\n"
            f"  prior_path: {train_paths['prior']}\n"
            f"  dataset: {train_paths['dataset']}\n"
            f"  save_file: {train_paths['save']}\n" + body + "\n"
        )

    def _run(self, runner, args):
        with patch("mach3sbitools.apps.train.InferenceHandler") as handler_cls:
            handler_cls.return_value = MagicMock()
            result = runner.invoke(cli, args)
            call = handler_cls.return_value.train_posterior.call_args
        return result, (call[0][0] if call else None)

    def test_values_come_from_the_file(self, runner, write_config, train_paths):
        path = self._train_config(
            write_config, train_paths, max_epochs=4321, batch_size=64
        )
        result, config = self._run(runner, ["-C", str(path), "train"])
        assert result.exit_code == 0, result.output
        assert config.max_epochs == 4321
        assert config.batch_size == 64

    def test_explicit_flag_beats_the_file(self, runner, write_config, train_paths):
        path = self._train_config(
            write_config, train_paths, max_epochs=4321, batch_size=64
        )
        _, config = self._run(runner, ["-C", str(path), "train", "--max_epochs", "7"])
        assert config.max_epochs == 7
        assert config.batch_size == 64

    def test_required_options_can_be_satisfied_by_the_file(
        self, runner, write_config, train_paths
    ):
        """-r/-d/-s are required flags; the file must be able to supply them."""
        path = self._train_config(write_config, train_paths)
        result, _ = self._run(runner, ["-C", str(path), "train"])
        assert result.exit_code == 0, result.output

    def test_bad_config_fails_before_running(self, runner, write_config):
        path = write_config("train:\n  hiden: 512\n")
        result = runner.invoke(cli, ["-C", str(path), "train"])
        assert result.exit_code != 0
        assert "hiden" in result.output

    def test_no_config_still_works(self, runner, train_paths):
        result, config = self._run(
            runner,
            [
                "train",
                "-r",
                str(train_paths["prior"]),
                "-d",
                str(train_paths["dataset"]),
                "-s",
                str(train_paths["save"]),
                "--max_epochs",
                "3",
            ],
        )
        assert result.exit_code == 0, result.output
        assert config.max_epochs == 3
