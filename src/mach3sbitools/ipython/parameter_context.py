"""
Shared logic for resolving simulator/prior parameter info for the dashboards.

The simulator's own parameter accessors (``get_parameter_names``,
``get_parameter_nominals``, ``get_parameter_bounds``) describe the *full*
theta vector the simulator expects. When a
:class:`~mach3sbitools.simulator.priors.prior.Prior` is built with
``nuisance_parameters``, only a subset of that vector is actually inferred
-- the excluded ("nuisance") entries are held fixed. The kept entries are
not necessarily contiguous (``nuisance_parameters`` is matched by fnmatch
pattern against arbitrary parameter names), so they can't be sliced out
with something like ``noms[-n:]``; they must be selected with the prior's
own boolean ``nuisance_filter`` mask.
"""

from dataclasses import dataclass

import torch

from mach3sbitools.simulator import Simulator
from mach3sbitools.utils import TorchDeviceHandler


@dataclass
class ParameterContext:
    """Resolved parameter metadata for building slider-driven dashboards."""

    parameter_names: list[str]
    nominal: list[float]
    lower_bounds: list[float]
    upper_bounds: list[float]
    nuisance_filter: torch.Tensor
    base_noms: torch.Tensor


def build_parameter_context(
    simulator: Simulator,
    device_handler: TorchDeviceHandler,
    base_noms: torch.Tensor | None = None,
) -> ParameterContext:
    """
    Resolve slider-facing parameter info from a simulator's prior.

    Uses ``simulator.prior.prior_data`` (already nuisance-filtered) for the
    names/nominal/bounds shown on sliders, and
    ``simulator.prior.nuisance_filter`` as the boolean mask for placing
    slider values back into the full-length vector the simulator expects.

    :param simulator: Configured :class:`~mach3sbitools.simulator.Simulator`.
    :param device_handler: Device handler for tensor conversion.
    :param base_noms: Full-length nominal parameter vector passed to
        ``simulator.simulator_wrapper.simulate``. Nuisance (filtered-out)
        entries stay fixed at these values. Defaults to the simulator's own
        (unfiltered) nominal vector.
    :returns: Resolved :class:`ParameterContext`.
    """
    wrapper = simulator.simulator_wrapper
    prior_data = simulator.prior.prior_data  # already nuisance-filtered
    nuisance_filter = simulator.prior.nuisance_filter.cpu()

    if base_noms is None:
        base_noms = device_handler.to_tensor(wrapper.get_parameter_nominals()).cpu()

    return ParameterContext(
        parameter_names=[str(p) for p in prior_data.parameter_names],
        nominal=prior_data.nominals.cpu().tolist(),
        lower_bounds=prior_data.lower_bounds.cpu().tolist(),
        upper_bounds=prior_data.upper_bounds.cpu().tolist(),
        nuisance_filter=nuisance_filter,
        base_noms=base_noms,
    )


def apply_sliders_to_noms(
    ctx: ParameterContext,
    slider_values: list[float],
    device_handler: TorchDeviceHandler,
) -> torch.Tensor:
    """
    Overwrite the inferred (non-nuisance) entries of ``ctx.base_noms``.

    :param ctx: Resolved parameter context.
    :param slider_values: Slider values, ordered to match
        ``ctx.parameter_names``.
    :param device_handler: Device handler for tensor conversion.
    :returns: Full-length parameter vector with inferred entries updated and
        nuisance entries left at their base values.
    """
    noms = ctx.base_noms.clone()
    noms[ctx.nuisance_filter] = device_handler.to_tensor(slider_values).cpu()
    return noms
