from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from mach3sbitools.inference import InferenceHandler
from mach3sbitools.simulator import Simulator


def normalise_logl(input_arr: np.ndarray) -> np.ndarray:
    """
    Standardise log-likelihoods so two differently-scaled sets overlay.

    :param input_arr: Raw log-likelihood values.
    :returns: Zero-mean, unit-variance values, or zeros if the input is
        constant.
    """
    mean = np.mean(input_arr)
    std_dev = np.std(input_arr)

    if std_dev == 0:
        return np.zeros_like(input_arr)

    return np.asarray((input_arr - mean) / std_dev)


def compare_logl(
    simulator: Simulator,
    inference_handler: InferenceHandler,
    n_samples: int,
    n_bins: int = 100,
    likelihood_range: tuple[float, float] | None = None,
    save_path: Path | None = None,
) -> None:
    """
    Overlay the log-likelihood of posterior samples against the simulator's.

    :param simulator: The simulator.
    :param inference_handler: Handler holding the trained posterior.
    :param n_samples: Number of samples to draw.
    :param n_bins: Histogram bins in the comparison plot.
    :param likelihood_range: ``(low, high)`` clip on the histogram range, or
        ``None`` to use the full data range.
    :param save_path: Where to save the figure. ``None`` skips saving.
    """

    simulator_samples = (
        inference_handler.sample_posterior(
            n_samples, simulator.simulator_wrapper.get_data_bins()
        )
        .cpu()
        .numpy()
    )

    sample_llh = np.array(
        [
            simulator.simulator_wrapper.get_log_likelihood(t)
            for t in tqdm(simulator_samples, desc="Get LLH from simulator")
        ]
    )
    normalised_sample = normalise_logl(sample_llh)

    x_data = simulator.simulator_wrapper.get_data_bins()

    inference_llh = (
        inference_handler.get_log_likelihood(simulator_samples, x_data).cpu().numpy()
    )
    normalised_inf = normalise_logl(inference_llh)

    fig, (ax2d, ax1d) = plt.subplots(nrows=2, ncols=1, figsize=(30, 30))

    # 2D log-l/log-l plot
    min_val = float(np.min([normalised_inf, normalised_sample]))
    max_val = float(np.max([normalised_inf, normalised_sample]))

    if likelihood_range is None:
        likelihood_range = (min_val, max_val)

    if likelihood_range[0] is None:
        likelihood_range[0] = min_val
    if likelihood_range[1] is None:
        likelihood_range[1] = max_val

    bins = np.linspace(likelihood_range[0], likelihood_range[1], n_bins)

    _, _, _, log_l2d = ax2d.hist2d(
        x=normalised_sample,
        y=normalised_inf,
        density=True,
        cmap="hot",
        bins=[bins, bins],
    )
    fig.colorbar(log_l2d, ax=ax2d)

    # y=x line
    ax2d.plot(
        likelihood_range, likelihood_range, color="white", linestyle="--", label="y=x"
    )

    # best fit line
    m, b = np.polyfit(normalised_sample, normalised_inf, 1)
    fit_x = np.array(likelihood_range)
    ax2d.plot(
        fit_x,
        m * fit_x + b,
        color="cyan",
        linestyle="--",
        label=f"Best fit (m={m:.2f}, b={b:.2f})",
    )

    ax2d.legend(loc="upper left")
    ax2d.set_xlabel("Model Likelihood (Normalised)")
    ax2d.set_ylabel("SBI Likelihood (Normalised)")

    # Project to 1D
    bins = np.linspace(min_val, max_val, 100)

    ax1d.hist(
        normalised_sample,
        bins=bins,
        label="Sample Likelihood (Normalised)",
        histtype="step",
        alpha=0.8,
        density=True,
    )
    ax1d.hist(
        normalised_inf,
        bins=bins,
        label="SBI Likelihood (Normalised)",
        histtype="step",
        alpha=0.8,
        density=True,
    )
    ax1d.legend(loc="upper right")

    if plt.isinteractive():
        fig.show()

    if save_path is not None:
        fig.savefig(f"{save_path.stem}_2D.{save_path.suffix}")
