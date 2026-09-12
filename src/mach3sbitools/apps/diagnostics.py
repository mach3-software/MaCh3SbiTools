from pathlib import Path
from tempfile import TemporaryDirectory

from mach3sbitools.diagnostics import SBCDiagnostic, compare_logl
from mach3sbitools.inference import InferenceHandler
from mach3sbitools.simulator import Simulator


def diagnostics_module(
    simulator_module: str,
    simulator_class: str,
    config: Path,
    posterior: Path,
    output_file: Path,
    nuisance_pars: list[str],
    cyclical_pars: list[str],
    flipped_pars: list[str],
    # Plot opts.
    make_sbc_rank: bool,
    make_expected_coverage: bool,
    make_tarp: bool,
    make_logl_comp: bool,
    n_prior_samples: int,
    n_posterior_samples: int,
) -> None:
    """
    Run posterior diagnostics and write the requested plots.

    :param simulator_module: Dotted module path holding the simulator class.
    :param simulator_class: Name of the simulator class within that module.
    :param config: Path to the simulator's configuration file.
    :param nuisance_pars: fnmatch patterns for parameters to exclude.
    :param cyclical_pars: fnmatch patterns for parameters using a cyclical prior.
    :param flipped_pars: fnmatch patterns for parameters that may flip sign.
    :param posterior: Path to a trained density estimator checkpoint.
    :param output_file: Directory the plots are written to.
    :param make_sbc_rank: Produce the SBC rank histogram.
    :param make_expected_coverage: Produce the expected-coverage plot.
    :param make_tarp: Produce the TARP coverage plot.
    :param make_logl_comp: Produce the log-likelihood comparison plot.
    :param n_prior_samples: Prior draws used to build the SBC sample set.
    :param n_posterior_samples: Posterior draws per prior sample.
    """
    # Set up simulator
    simulator = Simulator(
        simulator_module,
        simulator_class,
        config,
        nuisance_pars=nuisance_pars,
        cyclical_pars=cyclical_pars,
        flipped_pars=flipped_pars,
    )

    prior = simulator.prior

    output_file = Path(output_file)
    output_file.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as tmp_dir:
        prior_path = Path(tmp_dir) / "prior.pkl"
        prior.save(prior_path)
        inference_handler = InferenceHandler(prior_path)

    inference_handler.load_posterior(Path(posterior))

    if make_logl_comp:
        compare_logl(
            simulator,
            inference_handler,
            n_posterior_samples,
            save_path=output_file / "logl_comp.pdf",
        )

    if not make_sbc_rank and not make_expected_coverage and not make_tarp:
        return

    sbc_diag = SBCDiagnostic(simulator, inference_handler, output_file)

    sbc_diag.create_prior_samples(n_prior_samples)

    if make_sbc_rank:
        sbc_diag.rank_plot(n_posterior_samples)

    if make_expected_coverage:
        sbc_diag.expected_coverage(n_posterior_samples)

    if make_tarp:
        sbc_diag.tarp(n_posterior_samples)
