"""
Simulator injection utilities.

Simulators are expected to follow the :class:`SimulatorProtocol` contract and
be configurable via an input file (e.g. a MaCh3 fitter YAML). This module
handles dynamic import, protocol validation, and instantiation.
"""

import importlib
import inspect
import pkgutil
import sys
from collections.abc import Callable
from contextlib import contextmanager
from difflib import get_close_matches
from importlib.util import find_spec
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import numpy as np

from mach3sbitools.types import BoundaryConditions
from mach3sbitools.utils.logger import get_logger

logger = get_logger()


@contextmanager
def _with_cwd_on_path():
    """Temporarily add the caller's CWD to sys.path for local module resolution."""
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
        try:
            yield
        finally:
            sys.path.remove(cwd)
    else:
        yield


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SimulatorException(Exception):
    """Base exception for all simulator errors."""


class SimulatorImportError(SimulatorException):
    """Raised when the simulator module or class cannot be imported."""


class SimulatorImplementationError(SimulatorException):
    """Raised when a simulator class does not implement :class:`SimulatorProtocol`."""


class SimulatorSetupError(SimulatorException):
    """Raised when the simulator configuration file cannot be found."""


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SimulatorProtocol(Protocol):
    """
    Interface that every simulator must implement.

    Simulators are configured via a single file path passed to ``__init__``.
    For MaCh3 this is the fitter YAML config. All parameter-level methods
    operate over the full (un-filtered) parameter vector.
    """

    def __init__(self, simulator_config: Path | str) -> None:
        """
        Initialise and configure the simulator from a file.

        :param simulator_config: Path to the simulator configuration file.
        """
        ...

    def simulate(self, theta: list[float]) -> list[float]:
        """
        Run a single forward simulation.

        :param theta: Input parameter vector.
        :returns: Predicted observable vector *x*.
        """
        ...

    def get_parameter_names(self) -> list[str]:
        """
        Return the name of each parameter in *theta*.

        :returns: Ordered list of parameter name strings.
        """
        ...

    def get_parameter_bounds(self) -> BoundaryConditions:
        """
        Return hard lower and upper bounds for each parameter.

        :returns: Tuple of ``(lower_bounds, upper_bounds)``, each a list of
            floats with one entry per parameter.
        """
        ...

    def get_is_flat(self, i: int) -> bool:
        """
        Return whether parameter *i* should use a flat (uniform) prior.

        :param i: Zero-based parameter index.
        :returns: ``True`` if the parameter is flat, ``False`` for Gaussian.
        """
        ...

    def get_data_bins(self) -> list[float]:
        """
        Return the observed data bin values *x_o*.

        :returns: Observed data vector.
        """
        ...

    def get_parameter_nominals(self) -> list[float]:
        """
        Return the nominal (mean) value for each parameter.

        :returns: Ordered list of nominal values.
        """
        ...

    def get_parameter_errors(self) -> list[float]:
        """
        Return the 1σ error for each parameter.

        :returns: Ordered list of parameter errors.
        """
        ...

    def get_covariance_matrix(self) -> np.ndarray:
        """
        Return the full parameter covariance matrix.

        :returns: Square numpy array of shape ``(n_params, n_params)``.
        """
        ...

    def get_log_likelihood(self, theta: list[float]) -> float:
        """
        For a given theta value, returns the log-likelihood.

        :param theta: Parameter vector of length ``n_params``.
        :returns: Log-likelihood of the data given *theta*.
        """
        ...


def _implements(proto: type) -> Callable[[type], type]:
    """
    Build a class decorator asserting structural conformance to *proto*.

    :param proto: Protocol whose public methods must all be present.
    :returns: A decorator that validates the class it is applied to.
    """

    def _deco(cls_def):
        """
        Check *cls_def* implements every public method of the protocol.

        :param cls_def: The class being decorated.
        :returns: *cls_def*, unchanged.
        :raises TypeError: If any protocol method is missing.
        """
        proto_methods = {
            name
            for name, _ in inspect.getmembers(proto, predicate=inspect.isfunction)
            if not name.startswith("_")
        }
        missing = [m for m in proto_methods if not callable(getattr(cls_def, m, None))]

        if not missing:
            return cls_def

        raise SimulatorImplementationError(
            f"{cls_def.__name__} does not implement protocol {proto.__name__}.\n"
            f"Missing methods: {', '.join(sorted(missing))}\n"
            f"See {__file__} for the required interface."
        )

    return _deco


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _closest_match(name: str, candidates: list[str]) -> str | None:
    """
    Return the closest fuzzy match for *name* from *candidates*, or ``None``.

    :param name: The name to search for.
    :param candidates: List of candidate strings.
    :returns: Best match string, or ``None`` if no match above threshold.
    """
    matches = get_close_matches(name, candidates, n=1, cutoff=0.6)
    return matches[0] if matches else None


def _hint(name: str, candidates: list[str]) -> str:
    """
    Build a "did you mean?" hint string for error messages.

    :param name: The name that was not found.
    :param candidates: List of valid names to search.
    :returns: A hint string, or an empty string if no close match exists.
    """
    match = _closest_match(name, candidates)
    return f" Did you mean: {match}?" if match else ""


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def get_simulator(
    module_name: str, class_name: str, config: Path | str
) -> SimulatorProtocol:
    """
    Dynamically import, validate, and instantiate a simulator.

    The class is checked against :class:`SimulatorProtocol` before
    instantiation. Equivalent to::

        from <module_name> import <class_name>
        return class_name(config)

    .. code-block:: console

        # Example — loading a MaCh3 simulator
        get_simulator("mypackage.simulator", "MySimulator", Path("fitter.yaml"))

    :param module_name: Dotted Python module path (e.g. ``'mypackage.simulator'``).
    :param class_name: Name of the simulator class within the module.
    :param config: Path to the simulator configuration file.
    :returns: An instantiated, protocol-validated simulator object.
    :raises SimulatorImportError: If the module or class cannot be found.
    :raises SimulatorImplementationError: If the class does not satisfy
        :class:`SimulatorProtocol`.
    :raises SimulatorSetupError: If *config* does not exist on disk.
    """
    with _with_cwd_on_path():
        if find_spec(module_name) is None:
            installed = [m.name for m in pkgutil.iter_modules()]
            raise SimulatorImportError(
                f"Module '{module_name}' not found.{_hint(module_name, installed)}"
            )

        module = importlib.import_module(module_name)
    logger.info("Found simulator '%s'", module_name)

    if not hasattr(module, class_name):
        all_classes = [n for n, _ in inspect.getmembers(module, inspect.isclass)]
        raise SimulatorImportError(
            f"Class '{class_name}' not found in '{module_name}'."
            f"{_hint(class_name, all_classes)}"
        )

    simulator_cls = getattr(module, class_name)
    simulator_cls = _implements(SimulatorProtocol)(simulator_cls)
    logger.info("Imported simulator '%s' from '%s'", class_name, module_name)

    if not isinstance(config, Path):
        config = Path(config)

    if not config.exists():
        raise SimulatorSetupError(f"Config file not found: {config}")

    logger.info("Found simulator config '%s'", config)
    return cast(SimulatorProtocol, simulator_cls(str(config)))
