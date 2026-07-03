"""
Tests for mach3sbitools.ipython.widget_factories.

The dashboard builders (build_posterior_view, build_fluctuation_view) need
a live Simulator/InferenceHandler and a notebook front-end to do anything
meaningful, so they're exercised in the tutorial notebooks rather than
here. This covers the pure widget-construction logic instead.
"""

import ipywidgets as widgets

from mach3sbitools.ipython.widget_factories import (
    make_bins_slider,
    make_param_sliders,
    make_samples_slider,
    make_toggles,
    make_toys_slider,
)


class TestMakeParamSliders:
    def test_one_slider_per_parameter(self):
        sliders = make_param_sliders(
            parameter_names=["a", "b"],
            nominal=[0.0, 1.0],
            lower_bounds=[-5.0, -5.0],
            upper_bounds=[5.0, 5.0],
        )
        assert set(sliders) == {"a", "b"}
        assert all(isinstance(s, widgets.FloatSlider) for s in sliders.values())

    def test_nominal_value_used_as_starting_value(self):
        sliders = make_param_sliders(
            parameter_names=["a"],
            nominal=[2.5],
            lower_bounds=[-5.0],
            upper_bounds=[5.0],
        )
        assert sliders["a"].value == 2.5

    def test_slider_range_clamps_wide_bounds(self):
        sliders = make_param_sliders(
            parameter_names=["a"],
            nominal=[0.0],
            lower_bounds=[-1000.0],
            upper_bounds=[1000.0],
            slider_range=10.0,
        )
        assert sliders["a"].min == -10.0
        assert sliders["a"].max == 10.0

    def test_returns_fresh_instances_each_call(self):
        kwargs = dict(
            parameter_names=["a"],
            nominal=[0.0],
            lower_bounds=[-5.0],
            upper_bounds=[5.0],
        )
        first = make_param_sliders(**kwargs)
        second = make_param_sliders(**kwargs)
        assert first["a"] is not second["a"]


class TestScalarSliderFactories:
    def test_make_samples_slider_default(self):
        slider = make_samples_slider()
        assert isinstance(slider, widgets.IntSlider)
        assert slider.value == 100_000

    def test_make_toys_slider_default(self):
        slider = make_toys_slider()
        assert slider.value == 10

    def test_make_bins_slider_default(self):
        slider = make_bins_slider()
        assert slider.value == 100

    def test_scalar_sliders_accept_custom_value(self):
        assert make_samples_slider(value=5).value == 5
        assert make_toys_slider(value=3).value == 3
        assert make_bins_slider(value=50).value == 50


class TestMakeToggles:
    def test_returns_two_unchecked_checkboxes(self):
        hist_fill, minmax_toggle = make_toggles()
        assert isinstance(hist_fill, widgets.Checkbox)
        assert isinstance(minmax_toggle, widgets.Checkbox)
        assert hist_fill.value is False
        assert minmax_toggle.value is False

    def test_returns_fresh_instances_each_call(self):
        first, _ = make_toggles()
        second, _ = make_toggles()
        assert first is not second