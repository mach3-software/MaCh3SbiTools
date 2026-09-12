from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pyarrow import Table
from pyarrow import parquet as pq
from sbi.inference import ImportanceSamplingPosterior
from tqdm.asyncio import tqdm

from mach3sbitools.inference import InferenceHandler
from mach3sbitools.simulator import Simulator
from mach3sbitools.utils import get_device, get_logger, to_tensor


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
) -> None:
    """
    Reweight posterior samples against the simulator's true likelihood.

    :param simulator_module: Dotted module path holding the simulator class.
    :param simulator_class: Name of the simulator class within that module.
    :param config: Path to the simulator's configuration file.
    :param nuisance_pars: fnmatch patterns for parameters to exclude.
    :param cyclical_pars: fnmatch patterns for parameters using a cyclical prior.
    :param flipped_pars: fnmatch patterns for parameters that may flip sign.
    :param output_file: Destination parquet path for the weighted samples.
    :param n_samples: Number of samples to return.
    :param oversampling_factor: Proposals drawn per returned sample.
    :param max_sampling_batch: Largest proposal batch held in memory at once.
    :param posterior: Path to a trained density estimator checkpoint.
    """
    logger = get_logger()
    logger.info("Perform importance sampling")

    simulator = Simulator(
        simulator_module,
        simulator_class,
        config,
        nuisance_pars=nuisance_pars,
        cyclical_pars=cyclical_pars,
        flipped_pars=flipped_pars,
    )

    with TemporaryDirectory() as tmp_dir:
        prior_path = Path(tmp_dir) / "prior.pkl"
        simulator.prior.save(prior_path)
        inference_handler = InferenceHandler(prior_path)

    inference_handler.load_posterior(Path(posterior))
    inference_handler.build_posterior()

    if inference_handler.posterior is None:
        raise RuntimeError("No posterior found")

    def log_prob_fn(theta, _):
        """
        Evaluate the simulator's true log-likelihood for each proposal.

        :param theta: Proposals of shape ``(n, n_params)``.
        :param _: Observation argument required by sbi, unused here.
        :returns: Log-likelihood tensor of shape ``(n,)``.
        """
        return to_tensor(
            np.array(
                [
                    simulator.simulator_wrapper.get_log_likelihood(t)
                    for t in tqdm(theta.cpu().numpy())
                ]
            )
        )

    logger.info("Sampling...")

    xo = to_tensor(simulator.simulator_wrapper.get_data_bins())

    inference_handler.posterior.set_default_x(xo)

    posterior_sir = ImportanceSamplingPosterior(
        potential_fn=log_prob_fn,
        proposal=inference_handler.posterior,
        method="sir",
        device=get_device(),
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
