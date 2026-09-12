"""
YAML run configuration for the ``mach3sbi`` CLI.

A single file describes a whole study, one section per subcommand plus a
shared ``simulator`` block, and is turned into a click ``default_map``. Click
consults that map only for options the user did not type, so an explicit flag
always beats the file.

Example::

    logging:
      log_level: INFO
      log_file: run.log

    simulator:
      simulator_module: mypackage.simulator
      simulator_class: MySimulator
      config: fitter.yaml
      nuisance_pars: ["syst_*"]

    simulate:
      n_simulations: 100000
      output_file: sims/shard.feather

    train:
      prior_path: prior.pkl
      dataset: merged/
      save_file: models/run.ckpt
      hidden: 256
      max_epochs: 5000
"""

from pathlib import Path
from typing import Any

import click
import yaml

#: Section whose keys are offered to every subcommand that accepts them.
SHARED_SECTION = "simulator"

#: Section holding the top-level logging options.
LOGGING_SECTION = "logging"


class RunConfigError(Exception):
    """Raised when a run-configuration file cannot be used as written."""


def _normalise(name: str) -> str:
    """
    Fold the ``-``/``_`` difference in command names.

    :param name: A command or section name.
    :returns: The name with hyphens replaced by underscores.
    """
    return name.replace("-", "_")


def _accepted_options(command: click.Command) -> set[str]:
    """
    List the option names a command can actually be given.

    :param command: The click command to inspect.
    :returns: Option names, excluding click-option-group's internal
        group-title placeholders.
    """
    return {p.name for p in command.params if p.expose_value and p.name}


def _check_keys(values: Any, accepted: set[str], section: str) -> dict:
    """
    Validate one section against the options it is allowed to set.

    :param values: The section's parsed contents.
    :param accepted: Option names the section may set.
    :param section: Section name, for the error message.
    :returns: The section as a plain dict.
    :raises RunConfigError: If the section is not a mapping, or sets an
        option that does not exist.
    """
    if not isinstance(values, dict):
        raise RunConfigError(
            f"Section '{section}' must be a mapping, got {type(values).__name__}."
        )

    unknown = sorted(set(values) - accepted)
    if unknown:
        raise RunConfigError(
            f"Section '{section}' sets unknown option(s): {', '.join(unknown)}.\n"
            f"Valid options: {', '.join(sorted(accepted))}"
        )

    return dict(values)


def load_run_config(config_path: Path, group: click.Group) -> dict:
    """
    Read *config_path* into a click ``default_map`` for *group*.

    Every section is checked against the options it claims to set, so a typo
    fails immediately with the valid names rather than being ignored.

    :param config_path: Path to the YAML run configuration.
    :param group: The CLI group whose commands the sections must match.
    :returns: A nested ``default_map`` keyed by command name.
    :raises RunConfigError: If the file is malformed, names a section that is
        not a command, or sets an option that does not exist.
    """
    if not config_path.is_file():
        raise RunConfigError(f"Run config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise RunConfigError(
            f"{config_path} must contain a mapping of sections, "
            f"got {type(raw).__name__}."
        )

    by_normalised = {_normalise(name): name for name in group.commands}
    default_map: dict[str, Any] = {}

    for section, values in raw.items():
        if section == SHARED_SECTION:
            continue

        if section == LOGGING_SECTION:
            default_map.update(_check_keys(values, _accepted_options(group), section))
            continue

        command_name = by_normalised.get(_normalise(section))
        if command_name is None:
            raise RunConfigError(
                f"'{section}' is not a mach3sbi command.\n"
                f"Known commands: {', '.join(sorted(group.commands))}"
            )

        default_map[command_name] = _check_keys(
            values, _accepted_options(group.commands[command_name]), section
        )

    _merge_shared(raw.get(SHARED_SECTION) or {}, group, default_map)
    return default_map


def _merge_shared(shared: Any, group: click.Group, default_map: dict) -> None:
    """
    Offer the shared section to every command that accepts its keys.

    A command's own section wins over the shared block, and the shared block
    in turn is only a default — an explicit flag still beats both.

    :param shared: Contents of the ``simulator`` section.
    :param group: The CLI group being configured.
    :param default_map: Map to merge into, modified in place.
    :raises RunConfigError: If the section is not a mapping, or sets a key no
        command accepts.
    """
    if not isinstance(shared, dict):
        raise RunConfigError(
            f"Section '{SHARED_SECTION}' must be a mapping, "
            f"got {type(shared).__name__}."
        )
    if not shared:
        return

    every_option: set[str] = set()
    for name, command in group.commands.items():
        accepted = _accepted_options(command)
        every_option |= accepted

        applicable = {k: v for k, v in shared.items() if k in accepted}
        if applicable:
            default_map[name] = {**applicable, **default_map.get(name, {})}

    unusable = sorted(set(shared) - every_option)
    if unusable:
        raise RunConfigError(
            f"Section '{SHARED_SECTION}' sets option(s) no command accepts: "
            f"{', '.join(unusable)}"
        )
