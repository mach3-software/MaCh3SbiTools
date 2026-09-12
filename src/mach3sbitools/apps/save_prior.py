from pathlib import Path

from mach3sbitools.simulator import create_prior, get_simulator


def save_prior_module(
    simulator_module: str,
    simulator_class: str,
    config: Path,
    output_file: Path,
    nuisance_pars: list[str],
    cyclical_pars: list[str],
    flipped_pars: list[str],
) -> None:
    """Generate a Prior from a simulator and save it to disk.

    Instantiates the simulator, reads its parameter names, bounds, nominals,
    and covariance, then constructs and pickles a Prior object ready for use
    in training and inference.

    Example::

        mach3sbi create_prior \\
            -m mypackage.simulator -s MySimulator \\
            -c config.yaml -o prior.pkl

    :param simulator_module: Dotted module path holding the simulator class.
    :param simulator_class: Name of the simulator class within that module.
    :param config: Path to the simulator's configuration file.
    :param nuisance_pars: fnmatch patterns for parameters to exclude.
    :param cyclical_pars: fnmatch patterns for parameters using a cyclical prior.
    :param flipped_pars: fnmatch patterns for parameters that may flip sign.
    :param output_file: Destination path for the pickled prior.
    """
    injector = get_simulator(simulator_module, simulator_class, Path(config))
    prior = create_prior(injector, nuisance_pars, cyclical_pars, flipped_pars)
    prior.save(Path(output_file))
