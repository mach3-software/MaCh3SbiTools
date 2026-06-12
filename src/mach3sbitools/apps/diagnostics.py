from datetime import datetime
from pathlib import Path

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

    prior_path = Path(f"/tmp/{datetime.now()}_prior.pkl")
    prior.save(prior_path)

    inference_handler = InferenceHandler(prior_path, nuisance_pars)
    inference_handler.load_posterior(Path(posterior))

    output_file = Path(output_file)
    output_file.mkdir(parents=True, exist_ok=True)

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
