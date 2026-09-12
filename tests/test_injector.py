import pkgutil

import pytest

from mach3sbitools.simulator.simulator_injector import (
    SimulatorImplementationError,
    SimulatorImportError,
    SimulatorProtocol,
    _hint,
    get_simulator,
)
from mach3sbitools.utils import get_logger

logger = get_logger()


# Tests the injector
def test_import(simulator_module, simulator_class, dummy_config):
    simulator = get_simulator(simulator_module, simulator_class, dummy_config)
    assert isinstance(simulator, SimulatorProtocol)


def test_relative_import(simulator_module, simulator_class, dummy_config):
    # Check relative import
    simulator = get_simulator(
        f"{simulator_module}.dummy_simulator", simulator_class, dummy_config
    )
    assert isinstance(simulator, SimulatorProtocol)


def test_protocol_followed(simulator_module, dummy_config):
    # Checks a pre-built class that doesn't follow protocol
    bad_sim_class = "PoorlyDefinedSimulator"
    with pytest.raises(SimulatorImplementationError):
        get_simulator(simulator_module, bad_sim_class, dummy_config)


@pytest.mark.parametrize(
    "module,cls",
    [
        pytest.param("ABadPythonSimulator", "NotAClass", id="missing-module"),
        pytest.param("dummy_simulator", "NotAClass", id="missing-class"),
    ],
)
def test_import_failures(module, cls):
    with pytest.raises(SimulatorImportError):
        get_simulator(module, cls, "")


@pytest.mark.parametrize(
    "mangle",
    [
        pytest.param(str.capitalize, id="wrong-case"),
        pytest.param(lambda name: name[:-1] + "Z", id="typo-last-letter"),
    ],
)
def test_hint_suggests_the_real_module(simulator_module, mangle):
    installed = [m.name for m in pkgutil.iter_modules()]
    assert _hint(mangle(simulator_module), installed) == (
        f" Did you mean: {simulator_module}?"
    )


def test_hint_is_empty_when_nothing_is_close(simulator_module):
    assert _hint("zzzzzzzzzz", [simulator_module]) == ""


def test_import_error_message_carries_the_hint(simulator_module):
    with pytest.raises(SimulatorImportError, match="Did you mean"):
        get_simulator(simulator_module.capitalize(), "DummySimulator", "")
