"""
Interactive ipywidgets dashboards for exploring simulator/posterior
behaviour from within a Jupyter notebook.

::

    from mach3sbitools.ipython import build_posterior_view, build_fluctuation_view

    posterior_view = build_posterior_view(
        simulator=simulator,
        inference_handler=inference_handler,
        levels_dict={"1-Sigma Contour": 0.393, "2-Sigma Contour": 0.865},
    )

    fluctuation_view = build_fluctuation_view(
        simulator=simulator,
        inference_handler=inference_handler,
    )

Each ``build_*`` call creates its own independent widgets, wires up the
callbacks, displays the dashboard, and returns a dict of the created
widgets/state in case you want to inspect or drive them from the notebook.
See :func:`build_posterior_view` and :func:`build_fluctuation_view` for
details.
"""

from .fluctuation_dashboard import build_fluctuation_view
from .parameter_context import (
    ParameterContext,
    apply_sliders_to_noms,
    build_parameter_context,
)
from .posterior_dashboard import build_posterior_view
from .widget_factories import (
    make_bins_slider,
    make_param_sliders,
    make_samples_slider,
    make_toggles,
    make_toys_slider,
)

__all__ = [
    "ParameterContext",
    "apply_sliders_to_noms",
    "build_fluctuation_view",
    "build_parameter_context",
    "build_posterior_view",
    "make_bins_slider",
    "make_param_sliders",
    "make_samples_slider",
    "make_toggles",
    "make_toys_slider",
]