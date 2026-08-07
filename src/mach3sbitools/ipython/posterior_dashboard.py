"""Interactive posterior corner-plot dashboard."""

from typing import Any

import corner
import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
import torch
from IPython.display import clear_output, display
from matplotlib import pyplot as plt

from mach3sbitools.inference import InferenceHandler
from mach3sbitools.simulator import Simulator
from mach3sbitools.utils import TorchDeviceHandler

from .parameter_context import apply_sliders_to_noms, build_parameter_context
from .widget_factories import make_bins_slider, make_param_sliders, make_samples_slider


def build_posterior_view(
    *,
    simulator: Simulator,
    inference_handler: InferenceHandler,
    levels_dict: dict[str, float],
    device_handler: TorchDeviceHandler | None = None,
    base_noms: torch.Tensor | None = None,
) -> dict:
    """
    Build, wire up, and display an interactive posterior corner-plot dashboard.

    Parameter sliders control the entries of ``base_noms`` (or the entire
    vector, if ``base_noms`` is not given) selected by
    ``simulator.prior.nuisance_filter`` -- i.e. the non-nuisance, inferred
    parameters. The resulting vector is fed to the simulator to produce a
    new observation. Clicking "Run Inference" samples the posterior for
    that observation and caches the samples; moving the bins slider
    afterwards just redraws the cached samples with a new bin count,
    without resampling.

    :param simulator: Configured :class:`~mach3sbitools.simulator.Simulator`.
    :param inference_handler: Trained
        :class:`~mach3sbitools.inference.InferenceHandler`.
    :param levels_dict: Mapping of contour label to confidence level, e.g.
        ``{"1-Sigma Contour": 0.393, "2-Sigma Contour": 0.865}``. Scales to
        any number of levels.
    :param device_handler: Device handler for tensor conversion. Defaults to
        a fresh :class:`~mach3sbitools.utils.TorchDeviceHandler`.
    :param base_noms: Full-length nominal parameter vector passed to
        ``simulator.simulator_wrapper.simulate``. Slider values overwrite
        the entries selected by ``simulator.prior.nuisance_filter``;
        everything else stays fixed at ``base_noms``. Defaults to the
        simulator's own nominal vector (i.e. every parameter is
        slider-controlled).
    :returns: Dict of the created widgets/state (``sliders``,
        ``samples_slider``, ``bins_slider``, ``out``, ``results_cache``,
        ``run``), in case you want to inspect or drive them from the
        notebook.
    """
    device_handler = device_handler or TorchDeviceHandler()
    wrapper = simulator.simulator_wrapper
    ctx = build_parameter_context(simulator, device_handler, base_noms=base_noms)
    parameter_names = ctx.parameter_names

    sliders = make_param_sliders(
        ctx.parameter_names, ctx.nominal, ctx.lower_bounds, ctx.upper_bounds
    )
    samples_slider = make_samples_slider()
    bins_slider = make_bins_slider()
    out = widgets.Output()
    results_cache: dict = {}

    def _draw_corner(n_bins: int) -> None:
        plot_samples = results_cache["plot_samples"]
        noms = results_cache["noms"]

        with out:
            clear_output(wait=True)
            plt.close("all")

            sorted_levels = sorted(levels_dict.items(), key=lambda item: item[1])
            level_labels = [label for label, _ in sorted_levels]
            levels_values = [value for _, value in sorted_levels]
            num_levels = len(levels_values)
            cmap = mpl.colormaps["inferno"]
            level_colors = (
                [cmap(v) for v in np.linspace(0.3, 0.85, num_levels)]
                if num_levels > 1
                else [cmap(0.6)]
            )
            truth_color = cmap(0.1)

            fig = corner.corner(
                plot_samples,
                bins=n_bins,
                labels=parameter_names,
                truths=noms[ctx.nuisance_filter],
                truth_color=truth_color,
                fill_contours=False,
                levels=levels_values,
                color=level_colors[-1],
                contour_kwargs={"colors": level_colors, "linewidths": 1.5},
                contourf_kwargs={"colors": level_colors, "alpha": 0.5},
                smooth=1.0,
                smooth1d=1.0,
                plot_datapoints=False,
                plot_density=True,
            )

            legend_handles: list[Any] = [
                mpatches.Patch(
                    color=level_colors[num_levels - 1 - i],
                    alpha=0.7,
                    label=level_labels[i],
                )
                for i in range(num_levels)
            ]
            legend_handles.append(
                plt.Line2D([0], [0], color=truth_color, lw=2, label="True Value")
            )
            fig.legend(
                handles=legend_handles,
                loc="upper right",
                bbox_to_anchor=(0.85, 0.9),
                fontsize=11,
                frameon=True,
                facecolor="white",
                edgecolor="none",
            )

            display(fig)
            plt.close(fig)

    def redraw(_=None) -> None:
        if "plot_samples" not in results_cache:
            return
        _draw_corner(bins_slider.value)

    def run(_=None) -> None:
        slider_values = [sliders[name].value for name in parameter_names]
        noms = apply_sliders_to_noms(ctx, slider_values, device_handler)
        obs_new = wrapper.simulate(noms.tolist())
        current_n_samples = samples_slider.value

        posterior_samples = (
            inference_handler.sample_posterior(current_n_samples, x=obs_new)
            .cpu()
            .numpy()
        )
        max_plot_pts = current_n_samples // 10
        plot_samples = (
            posterior_samples[
                np.random.choice(len(posterior_samples), max_plot_pts, replace=False)
            ]
            if len(posterior_samples) > max_plot_pts
            else posterior_samples
        )

        results_cache["posterior_samples"] = posterior_samples
        results_cache["plot_samples"] = plot_samples
        results_cache["noms"] = noms

        _draw_corner(bins_slider.value)

    bins_slider.observe(redraw, names="value")

    run_button = widgets.Button(
        description="Run Inference", button_style="success", icon="play"
    )
    run_button.on_click(run)

    ui = widgets.VBox(
        [
            *list(sliders.values()),
            widgets.HTML("<hr>"),
            samples_slider,
            bins_slider,
            run_button,
        ]
    )
    display(ui, out)

    return {
        "sliders": sliders,
        "samples_slider": samples_slider,
        "bins_slider": bins_slider,
        "out": out,
        "results_cache": results_cache,
        "run": run,
    }
