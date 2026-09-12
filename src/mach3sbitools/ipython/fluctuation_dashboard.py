"""Interactive Poisson-fluctuation toy-histogram dashboard."""

import ipywidgets as widgets
import numpy as np
import torch
from IPython.display import clear_output, display
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from mach3sbitools.inference import InferenceHandler
from mach3sbitools.simulator import Simulator
from mach3sbitools.utils import to_tensor

from .parameter_context import apply_sliders_to_noms, build_parameter_context
from .widget_factories import (
    make_bins_slider,
    make_param_sliders,
    make_samples_slider,
    make_toggles,
    make_toys_slider,
)


def build_fluctuation_view(
    *,
    simulator: Simulator,
    inference_handler: InferenceHandler,
    base_noms: torch.Tensor | None = None,
) -> dict:
    """
    Build, wire up, and display a Poisson-fluctuation toy-histogram dashboard.

    Clicking "Run Inference" draws ``N Toys`` Poisson fluctuations of the
    simulated observation, samples the posterior for each, and caches the
    raw samples. The bins / fill / min-max controls afterwards just redraw
    from the cache, without re-running inference.

    Parameters mirror
    :func:`~mach3sbitools.ipython.posterior_dashboard.build_posterior_view`;
    this view creates its own independent slider widgets, so it can be shown
    alongside a posterior view without their controls interfering with each
    other.

    :param simulator: Configured :class:`~mach3sbitools.simulator.Simulator`.
    :param inference_handler: Trained
        :class:`~mach3sbitools.inference.InferenceHandler`.
    :param base_noms: Full-length nominal parameter vector passed to
        ``simulator.simulator_wrapper.simulate``. Slider values overwrite
        the entries selected by ``simulator.prior.nuisance_filter`` (i.e.
        the non-nuisance, inferred parameters); everything else stays fixed
        at ``base_noms``. Defaults to the simulator's own nominal vector.
    :returns: Dict of the created widgets/state.
    """
    wrapper = simulator.simulator_wrapper
    ctx = build_parameter_context(simulator, base_noms=base_noms)
    parameter_names = ctx.parameter_names
    n_params = len(parameter_names)

    sliders = make_param_sliders(
        ctx.parameter_names, ctx.nominal, ctx.lower_bounds, ctx.upper_bounds
    )
    samples_slider = make_samples_slider()
    toys_slider = make_toys_slider()
    bins_slider = make_bins_slider()
    hist_fill, minmax_toggle = make_toggles()
    out = widgets.Output()
    results_cache: dict = {}

    def fluctuate(x_obs):
        """
        Apply Poisson noise to an observation.

        :param x_obs: Expected bin contents.
        :returns: A Poisson-fluctuated toy observation.
        """
        return to_tensor([np.random.poisson(b) for b in x_obs])

    def _plot_param(
        i: int,
        posterior_samples: np.ndarray,
        nominal_samps: np.ndarray,
        n_bins: int,
        fill: bool,
        show_minmax: bool,
    ) -> None:
        """
        Draw the toy-spread histogram for one parameter.

        :param i: Parameter index.
        :param posterior_samples: Samples of shape ``(n_toys, n_samples, n_params)``.
        :param nominal_samps: Samples drawn at the nominal observation.
        :param n_bins: Histogram bins.
        :param fill: Fill the histograms rather than drawing outlines.
        :param show_minmax: Shade the per-bin min/max envelope across toys.
        """
        minmax_color = "#8f8882"
        n_toys = posterior_samples.shape[0]

        fig, ax = plt.subplots()
        bin_edges = np.histogram_bin_edges(
            posterior_samples[:, :, i].ravel(), bins=n_bins
        )

        # Histogram every toy once, up front. Reused for both the per-toy
        # draw and the min/max envelope below, instead of re-binning twice.
        counts = np.array(
            [
                np.histogram(samp[:, i], bins=bin_edges, density=True)[0]
                for samp in posterior_samples
            ]
        )  # shape: (n_toys, n_bins)

        # ax.stairs draws one lightweight PolyCollection per toy, instead of
        # ax.hist(histtype='bar')'s one Patch per bin -- much cheaper once
        # n_toys * n_bins gets large.
        for c in counts:
            ax.stairs(
                c,
                bin_edges,
                fill=fill,
                baseline=0 if fill else None,
                alpha=1 / n_toys,
                color="orange",
                linewidth=0 if fill else 1,
            )

        if show_minmax:
            ax.stairs(
                counts.min(axis=0),
                bin_edges,
                color=minmax_color,
                linewidth=1,
                baseline=None,
                linestyle="--",
            )
            ax.stairs(
                counts.max(axis=0),
                bin_edges,
                color=minmax_color,
                linewidth=1,
                baseline=None,
                label="Toy min/max",
                linestyle="--",
            )

        ax.hist(
            nominal_samps[:, i],
            bins=bin_edges.tolist(),
            color="k",
            histtype="step",
            density=True,
            label="Nominal",
        )
        ax.legend(loc="upper right")
        ax.set_title(parameter_names[i])
        ax.set_xlabel(parameter_names[i])
        ax.set_ylabel("Count")

        display(fig)
        plt.close(fig)

    def redraw(_=None) -> None:
        """
        Re-render the cached results after a display-only control changes.

        :param _: Widget change event, unused.
        """
        if "posterior_samples" not in results_cache:
            return
        with out:
            clear_output(wait=True)
            plt.close("all")
            for i in range(n_params):
                _plot_param(
                    i,
                    results_cache["posterior_samples"],
                    results_cache["nominal_samps"],
                    bins_slider.value,
                    hist_fill.value,
                    minmax_toggle.value,
                )

    def run_fluctuate(_=None) -> None:
        """
        Re-simulate, re-sample the posterior, and redraw.

        :param _: Widget change event, unused.
        """
        slider_values = [sliders[name].value for name in parameter_names]
        noms = apply_sliders_to_noms(ctx, slider_values)
        obs_new = wrapper.simulate(noms.tolist())

        current_n_samples = samples_slider.value
        n_toys = toys_slider.value

        with out:
            clear_output(wait=True)
            posterior_samples = np.array(
                [
                    inference_handler.sample_posterior(
                        current_n_samples,
                        x=fluctuate(obs_new),
                        show_progress_bars=False,
                    )
                    .cpu()
                    .numpy()
                    for _ in tqdm(range(n_toys))
                ]
            )
            nominal_samps = (
                inference_handler.sample_posterior(
                    current_n_samples, x=obs_new, show_progress_bars=False
                )
                .cpu()
                .numpy()
            )

        results_cache["posterior_samples"] = posterior_samples
        results_cache["nominal_samps"] = nominal_samps
        redraw()

    bins_slider.observe(redraw, names="value")
    hist_fill.observe(redraw, names="value")
    minmax_toggle.observe(redraw, names="value")

    run_button = widgets.Button(
        description="Run Inference", button_style="success", icon="play"
    )
    run_button.on_click(run_fluctuate)

    ui = widgets.VBox(
        [
            *list(sliders.values()),
            widgets.HTML("<hr>"),
            samples_slider,
            toys_slider,
            bins_slider,
            minmax_toggle,
            hist_fill,
            widgets.HTML("<hr>"),
            run_button,
        ]
    )
    display(ui, out)

    return {
        "sliders": sliders,
        "samples_slider": samples_slider,
        "toys_slider": toys_slider,
        "bins_slider": bins_slider,
        "hist_fill": hist_fill,
        "minmax_toggle": minmax_toggle,
        "out": out,
        "results_cache": results_cache,
        "run_fluctuate": run_fluctuate,
    }
