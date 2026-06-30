from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pyarrow import Table
from pyarrow import parquet as pq
from sbi.analysis import pairplot

from mach3sbitools.inference import InferenceHandler
from mach3sbitools.utils import get_logger


def inference(
    posterior: Path,
    prior_path: Path,
    save_file: Path,
    n_samples: int,
    observed_data_file: Path,
    nuisance_pars: list[str],
) -> None:
    """Sample the posterior distribution conditioned on observed data.x

    Loads a trained density estimator checkpoint, reads the model architecture
    directly from it (no ``--model`` / ``--hidden`` / etc. flags required),
    conditions the posterior on the observed data vector, draws
    ``--n_samples`` samples, and writes them as a parquet file with one column
    per parameter.

    Example::

        mach3sbi inference \\
            -i models/best.pt -r prior.pkl \\
            -n 100000 -o observed.parquet -s samples.parquet
    """
    logger = get_logger()

    if not isinstance(save_file, Path):
        save_file = Path(save_file)

    if save_file.is_file():
        logger.warning("Found %s, deleting", save_file)
        save_file.unlink()

    save_file.parent.mkdir(parents=True, exist_ok=True)

    logger = get_logger()

    # PosteriorConfig is recovered from the checkpoint — the caller does not
    # need to supply (and cannot accidentally mismatch) architecture flags.
    inference_handler = InferenceHandler(Path(prior_path))
    inference_handler.load_posterior(Path(posterior))

    parameter_names = inference_handler.prior.prior_data.parameter_names
    logger.info(parameter_names)
    observed_data = np.array(pq.read_table(observed_data_file)["data"])

    samples = inference_handler.sample_posterior(n_samples, observed_data).cpu().numpy()

    pairplot(samples, labels=[[p] for p in parameter_names])
    plt.savefig(save_file.with_suffix(".pdf"))

    data_table = Table.from_pydict(
        {p: samples[:, i] for i, p in enumerate(parameter_names)}
    )
    pq.write_table(data_table, save_file)
    logger.info(f"Saved to {save_file}")
