"""Factory functions for building independent ipywidgets controls.

Each function returns brand-new widget instances on every call, so that
multiple dashboards (e.g. a posterior view and a fluctuation view shown
side by side) never share slider/toggle state with one another.
"""

import ipywidgets as widgets


def make_param_sliders(
    parameter_names: list[str],
    nominal: list[float],
    lower_bounds: list[float],
    upper_bounds: list[float],
    slider_range: float = 10.0,
) -> dict[str, widgets.FloatSlider]:
    """
    Build one :class:`~ipywidgets.FloatSlider` per parameter.

    :param parameter_names: Ordered parameter names.
    :param nominal: Nominal (starting) value for each parameter.
    :param lower_bounds: Hard lower bound for each parameter.
    :param upper_bounds: Hard upper bound for each parameter.
    :param slider_range: Clamp each slider's min/max to within
        ``+/- slider_range`` of zero, on top of the hard bounds, so sliders
        stay usable even for parameters with very wide priors.
    :returns: Mapping of parameter name to its slider widget.
    """
    sliders: dict[str, widgets.FloatSlider] = {}
    for i, name in enumerate(parameter_names):
        lo = max(-slider_range, lower_bounds[i])
        hi = min(slider_range, upper_bounds[i])
        sliders[name] = widgets.FloatSlider(
            value=nominal[i],
            min=lo,
            max=hi,
            step=abs(hi - lo) / 100,
            description=name,
            continuous_update=False,
            readout_format=".4e",
        )
    return sliders


def make_samples_slider(value: int = 100_000) -> widgets.IntSlider:
    """Build an ``N Samples`` slider controlling posterior sample count."""
    return widgets.IntSlider(
        value=value,
        min=1,
        max=10_000_000,
        step=10_000,
        description="N Samples",
        continuous_update=False,
        style={"description_width": "initial"},
    )


def make_toys_slider(value: int = 10) -> widgets.IntSlider:
    """Build an ``N Toys`` slider controlling Poisson-fluctuation toy count."""
    return widgets.IntSlider(
        value=value,
        min=1,
        max=10000,
        step=1,
        description="N Toys",
        continuous_update=False,
        style={"description_width": "initial"},
    )


def make_bins_slider(value: int = 100) -> widgets.IntSlider:
    """Build an ``N Bins`` slider controlling histogram/corner-plot binning."""
    return widgets.IntSlider(
        value=value,
        min=1,
        max=1000,
        step=1,
        description="N Bins",
        continuous_update=False,
        style={"description_width": "initial"},
    )


def make_toggles() -> tuple[widgets.Checkbox, widgets.Checkbox]:
    """Build the ``Fill Hists?`` and ``Show Min/Max?`` checkboxes."""
    hist_fill = widgets.Checkbox(value=False, description="Fill Hists?", indent=False)
    minmax_toggle = widgets.Checkbox(
        value=False, description="Show Min/Max?", indent=False
    )
    return hist_fill, minmax_toggle
