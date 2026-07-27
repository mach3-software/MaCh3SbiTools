from pathlib import Path

from mach3sbitools.simulator import Simulator


def save_data_module(
    simulator_module: str,
    simulator_class: str,
    config: Path,
    output_file: Path,
    nuisance_pars: list[str],
    cyclical_pars: list[str],
    flipped_pars: list[str],
) -> None:
    """Extract and save the observed data bins from the simulator.

    Calls ``get_data_bins()`` on the simulator and writes the result to a
    parquet file. Useful for producing the observed data vector ``x_o``
    that is passed to ``inference``.

    Example::

        mach3sbi save_data \\
            -m mypackage.simulator -s MySimulator \\
            -c config.yaml -o observed.parquet
    """
    simulator = Simulator(
        simulator_module,
        simulator_class,
        Path(config),
        nuisance_pars=nuisance_pars,
        cyclical_pars=cyclical_pars,
        flipped_pars=flipped_pars,
    )
    simulator.save_data(Path(output_file))
3