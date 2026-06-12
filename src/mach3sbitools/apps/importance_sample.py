from pathlib import Path

import numpy as np
from pyarrow import Table
from pyarrow import parquet as pq
from sbi.inference import ImportanceSamplingPosterior
from tqdm.asyncio import tqdm

from mach3sbitools.inference import InferenceHandler
from mach3sbitools.simulator import Simulator
from mach3sbitools.utils import TorchDeviceHandler, get_logger


def importance_sample_module(
    simulator_module: str,
    simulator_class: str,
    config: Path,
    output_file: Path,
    n_samples: int,
    oversampling_factor: int,
    max_sampling_batch: int,
    posterior: Path,
    nuisance_pars: list[str],
    cyclical_pars: list[str],
    flipped_pars: list[str],
):
    logger = get_logger()
    logger.info("Perform importance sampling")

    device_handler = TorchDeviceHandler()
    """Sample the posterior distribution conditioned on observed data."""
    simulator = Simulator(
        simulator_module,
        simulator_class,
        config,
        nuisance_pars=nuisance_pars,
        cyclical_pars=cyclical_pars,
        flipped_pars=flipped_pars,
    )

    prior_path = Path("/tmp/prior.pkl")
    simulator.prior.save(prior_path)
    inference_handler = InferenceHandler(Path(prior_path), nuisance_pars)
    inference_handler.load_posterior(Path(posterior))
    inference_handler.build_posterior()

    if inference_handler.posterior is None:
        raise RuntimeError("No posterior found")

    def log_prob_fn(theta, _):
        return device_handler.to_tensor(
            np.array(
                [
                    simulator.simulator_wrapper.get_log_likelihood(t)
                    for t in tqdm(theta.cpu().numpy())
                ]
            )
        )

    logger.info("Sampling...")

    xo = device_handler.to_tensor(simulator.simulator_wrapper.get_data_bins())

    inference_handler.posterior.set_default_x(xo)

    posterior_sir = ImportanceSamplingPosterior(
        potential_fn=log_prob_fn,
        proposal=inference_handler.posterior,
        method="sir",
        device=device_handler.device,
    )

    theta_inferred = posterior_sir.sample(
        (n_samples,),
        oversampling_factor=oversampling_factor,
        max_sampling_batch_size=max_sampling_batch,
        x=xo,
        show_progress_bars=True,
    )
    parameter_names = inference_handler.prior.prior_data.parameter_names
    data_table = Table.from_pydict(
        {p: theta_inferred[:, i] for i, p in enumerate(parameter_names)}
    )
    pq.write_table(data_table, output_file)
    logger.info(f"Saved to {output_file}")
