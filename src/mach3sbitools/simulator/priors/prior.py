"""
Prior distribution for MaCh3 SBI.

Constructs a composite prior from four distribution types, checked in order:

1. **Cyclical** — parameters matching *cyclical_parameters* patterns, forced
   to bounds of ``[-2π, 2π]``.
2. **Flipped Uniform** — parameters matching *flipped_parameters* patterns.
   Support is ``[lower, upper] + [-upper, -lower]`` where ``lower``/``upper``
   are read from the parameter's existing bounds (which must be positive and
   satisfy ``0 < lower < upper``).
3. **Flat (Uniform)** — parameters flagged via *flat_msk* and not cyclical or
   flipped.
4. **Gaussian** — all remaining parameters, modelled as a
   :class:`~torch.distributions.MultivariateNormal`.
"""

import fnmatch
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import numpy as np
import torch
from torch.distributions import Uniform, constraints

from mach3sbitools.utils import TorchDeviceHandler, get_logger

from ..simulator_injector import SimulatorProtocol
from .cyclical_distribution import CyclicalDistribution
from .dataclasses import PriorData
from .flipped_uniform_distribution import FlippedUniformDistribution
from .truncated_gaussian_distribution import TruncatedGaussianDistribution

size_: TypeAlias = torch.Size | list[int] | tuple[int, ...]


class PriorNotFound(Exception):
    """Raised when a prior file cannot be found or deserialised."""


logger = get_logger()


@dataclass(frozen=True)
class MaskDistributionMap:
    """
    Associates a boolean parameter mask with its distribution.

    :param mask: Boolean tensor of shape ``(n_params,)`` selecting the
        parameters governed by *distribution*.
    :param distribution: The :class:`torch.distributions.Distribution`
        for the selected parameters.
    """

    mask: torch.Tensor
    distribution: torch.distributions.Distribution

    def to(self, device: torch.device | str) -> "MaskDistributionMap":
        """
        Move *mask* to *device* (distribution tensors are not moved).

        :param device: Target PyTorch device.
        :returns: New :class:`MaskDistributionMap` with mask on *device*.
        """
        return MaskDistributionMap(
            mask=self.mask.to(device), distribution=self.distribution
        )


class Prior(torch.distributions.Distribution):
    """
    Composite MaCh3 prior combining cyclical, flipped-uniform, flat, and
    Gaussian components.

    Designed to replicate MaCh3's prior construction
    (https://github.com/mach3-software/MaCh3) and satisfy the ``sbi``
    :class:`~torch.distributions.Distribution` interface.

    Parameters are assigned to distributions in the following order:

    - **Cyclical** — matched by *cyclical_parameters* (fnmatch patterns).
    - **Flipped Uniform** — matched by *flipped_parameters* (fnmatch patterns).
      Each matched parameter must have positive bounds
      ``0 < lower_bound < upper_bound``; its support becomes
      ``[lower, upper] + [-upper, -lower]``.
    - **Flat** — flagged by *flat_msk* and not cyclical or flipped.
    - **Gaussian** — everything else.

    .. warning::

        The nuisance filter is fixed at construction time. Calling
        :meth:`set_nuisance_filter` after construction is not supported —
        the distribution masks are built once against the filtered parameter
        set and cannot be safely remapped afterwards. To change the nuisance
        filter, construct a new :class:`Prior`.
    """

    def __init__(
        self,
        prior_data: PriorData,
        flat_msk: list[bool] | None = None,
        cyclical_parameters: list[str] | None = None,
        nuisance_parameters: list[str] | None = None,
        flipped_parameters: list[str] | None = None,
    ):
        """
        Construct the composite prior.

        :param prior_data: Raw prior arrays (names, nominals, bounds, covariance).
        :param flat_msk: Per-parameter flat flags (index-aligned with
            *prior_data*). Defaults to all ``False`` if not provided.
        :param cyclical_parameters: fnmatch patterns selecting cyclical
            parameters. Matched parameters receive bounds of ``±2π``.
        :param nuisance_parameters: fnmatch patterns selecting parameters to
            exclude. Fixed at construction time — cannot be changed later.
        :param flipped_parameters: fnmatch patterns selecting parameters that
            use a bimodal uniform prior over
            ``[lower, upper] + [-upper, -lower]``.  The bounds are read from
            *prior_data* and must satisfy ``0 < lower < upper``.
        """
        self.device_handler = TorchDeviceHandler()
        self._prior_data = prior_data

        # Apply nuisance filter once — masks are built against the filtered set
        # and cannot be safely remapped if the filter changes afterwards.
        self._nuisance_filter = self._build_nuisance_filter(nuisance_parameters)
        self._priors: list[MaskDistributionMap] = []

        n_params = len(self.prior_data.nominals)

        # ── Cyclical mask ──────────────────────────────────────────────────
        if cyclical_parameters:
            cyclical_mask_ = [
                any(fnmatch.fnmatch(p, c) for c in cyclical_parameters)
                for p in self.prior_data.parameter_names
            ]
            cyclical_mask = self.device_handler.to_tensor(cyclical_mask_)
        else:
            cyclical_mask = torch.zeros(
                n_params, dtype=torch.bool, device=self.device_handler.device
            )

        if any(cyclical_mask):
            self._prior_data[self._nuisance_filter].lower_bounds[cyclical_mask] = (
                -2 * torch.pi
            )
            self._prior_data[self._nuisance_filter].upper_bounds[cyclical_mask] = (
                2 * torch.pi
            )
            self._priors.append(self._get_cyclical_map(cyclical_mask))

        # ── Flipped-uniform mask ───────────────────────────────────────────
        if flipped_parameters:
            flipped_mask_ = [
                any(fnmatch.fnmatch(p, f) for f in flipped_parameters)
                for p in self.prior_data.parameter_names
            ]
            flipped_mask = self.device_handler.to_tensor(flipped_mask_).bool()
            # Flipped params must not also be cyclical
            flipped_mask = flipped_mask & ~cyclical_mask
        else:
            flipped_mask = torch.zeros(
                n_params, dtype=torch.bool, device=self.device_handler.device
            )

        # Store for use in check_bounds — flipped params are valid in EITHER
        # region so the standard lower/upper bounds check would incorrectly
        # reject samples drawn from the negative region.
        self._flipped_mask = flipped_mask

        if any(flipped_mask):
            self._priors.extend(self._get_flipped_maps(flipped_mask))

        # ── Flat mask ──────────────────────────────────────────────────────
        # Guard against None so the tensor conversion doesn't crash.
        flat_msk = flat_msk if flat_msk is not None else [False] * n_params
        flat_mask = (
            self.device_handler.to_tensor(flat_msk).bool()
            & ~cyclical_mask
            & ~flipped_mask
        )
        if any(flat_mask):
            self._priors.append(self._get_flat_map(flat_mask))

        # ── Gaussian mask ──────────────────────────────────────────────────
        gaussian_mask = ~cyclical_mask & ~flipped_mask & ~flat_mask
        if any(gaussian_mask):
            self._priors.append(self._get_gaussian_map(gaussian_mask))

        super().__init__(
            batch_shape=torch.Size(),
            event_shape=torch.Size([n_params]),
            validate_args=False,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_nuisance_filter(
        self, nuisance_patterns: list[str] | None
    ) -> torch.Tensor:
        """
        Build a boolean keep-mask from *nuisance_patterns*.

        :param nuisance_patterns: fnmatch patterns, or ``None`` to keep all.
        :returns: Boolean tensor of shape ``(n_all_params,)``.
        """
        if nuisance_patterns is None:
            n_pars = len(self._prior_data.parameter_names)
            return torch.ones(
                n_pars, dtype=torch.bool, device=self.device_handler.device
            )

        keep = [
            not any(fnmatch.fnmatch(p, n) for n in nuisance_patterns)
            for p in self._prior_data.parameter_names
        ]
        return self.device_handler.to_tensor(keep)

    # ── Private distribution builders ──────────────────────────────────────────

    def _get_cyclical_map(self, cyclical_mask: torch.Tensor) -> MaskDistributionMap:
        cyclical_data = self.prior_data[cyclical_mask]
        cyclical_dist = CyclicalDistribution(cyclical_data.nominals)
        return MaskDistributionMap(cyclical_mask, cyclical_dist)

    def _get_flipped_maps(
        self, flipped_mask: torch.Tensor
    ) -> list[MaskDistributionMap]:
        """
        Build one :class:`MaskDistributionMap` per flipped parameter.

        Each flipped parameter may have *different* ``lower``/``upper`` bounds,
        so a separate :class:`~.FlippedUniformDistribution` is created for each
        one.  The individual single-parameter masks are disjoint and together
        cover exactly the bits set in *flipped_mask*.

        :param flipped_mask: Boolean tensor of shape ``(n_params,)`` with
            ``True`` for every flipped parameter.
        :returns: List of :class:`MaskDistributionMap` objects, one per
            flipped parameter.
        :raises ValueError: If any matched parameter has bounds that violate
            ``0 < lower < upper``.
        """
        maps: list[MaskDistributionMap] = []
        flipped_indices = flipped_mask.nonzero(as_tuple=True)[0]

        for idx in flipped_indices:
            # Build a single-parameter mask
            single_mask = torch.zeros(
                len(flipped_mask), dtype=torch.bool, device=self.device_handler.device
            )
            single_mask[idx] = True

            lower_val = float(self.prior_data.lower_bounds[idx].item())
            upper_val = float(self.prior_data.upper_bounds[idx].item())
            param_name = self.prior_data.parameter_names[idx.item()]

            if lower_val <= 0 or upper_val <= lower_val:
                raise ValueError(
                    f"Flipped parameter '{param_name}' has bounds "
                    f"[{lower_val}, {upper_val}] but FlippedUniformDistribution "
                    f"requires 0 < lower < upper.  "
                    f"Set positive bounds in your simulator config."
                )

            single_data = self.prior_data[single_mask]
            dist = FlippedUniformDistribution(
                nominals=single_data.nominals,
                lower=lower_val,
                upper=upper_val,
            )
            maps.append(MaskDistributionMap(single_mask, dist))

        return maps

    def _get_flat_map(self, flat_mask: torch.Tensor) -> MaskDistributionMap:
        flat_data = self.prior_data[flat_mask]
        flat_dist = Uniform(flat_data.lower_bounds, flat_data.upper_bounds)
        return MaskDistributionMap(flat_mask, flat_dist)

    def _get_gaussian_map(self, gaussian_mask: torch.Tensor) -> MaskDistributionMap:
        gaussian_data = self.prior_data[gaussian_mask]
        dist = TruncatedGaussianDistribution(
            mean=gaussian_data.nominals,
            covariance=gaussian_data.covariance_matrix,
            lower_bounds=gaussian_data.lower_bounds,
            upper_bounds=gaussian_data.upper_bounds,
        )
        return MaskDistributionMap(gaussian_mask, dist)

    # ── Properties ─────────────────────────────────────────────────────────────
    @property
    def effective_lower_bounds(self) -> torch.Tensor:
        """Lower bounds accounting for flipped parameters (which are valid in [-upper, -lower])."""
        lb = self.prior_data.lower_bounds.clone()
        if self._flipped_mask.any():
            # Flipped params are valid from -upper to +upper (excluding the gap)
            lb[self._flipped_mask] = -self.prior_data.upper_bounds[self._flipped_mask]
        return lb

    @property
    def effective_upper_bounds(self) -> torch.Tensor:
        """Upper bounds accounting for flipped parameters."""
        return self.prior_data.upper_bounds

    @property
    def prior_data(self) -> PriorData:
        """Active :class:`PriorData` after applying the nuisance filter."""
        return self._prior_data[self._nuisance_filter]

    @property
    def mean(self) -> torch.Tensor:
        """Prior mean — the nominal parameter values."""
        return self.device_handler.to_tensor(self.prior_data.nominals)

    @property
    def n_params(self) -> int:
        """Number of active (non-nuisance) parameters."""
        return len(self.prior_data.nominals)

    @property
    def variance(self) -> torch.Tensor:
        """
        Per-parameter prior variance, assembled from all sub-distributions.

        The mask for each sub-distribution is sized against the filtered
        parameter set (same as the tensor being filled), so shapes are always
        consistent.
        """
        variance = torch.zeros(self.n_params, device=self.device_handler.device)
        for mask_map in self._priors:
            variance[mask_map.mask] = mask_map.distribution.variance
        return variance

    @property
    def support(self):
        class CheckBoundsConstraint(constraints.Constraint):
            is_discrete = False
            event_dim = 1

            def __init__(self_, prior):
                self_._prior = prior
                super().__init__()

            def check(self_, value: torch.Tensor) -> torch.Tensor:
                # check_bounds expects (n_samples, n_params) so unsqueeze if needed
                if value.dim() == 1:
                    return cast(
                        torch.Tensor,
                        self_._prior.check_bounds(value.unsqueeze(0)).squeeze(0),
                    )
                return cast(torch.Tensor, self_._prior.check_bounds(value))

        return CheckBoundsConstraint(self)

    # ── Distribution interface ─────────────────────────────────────────────────

    def sample(self, sample_shape=torch.Size([])) -> torch.Tensor:
        """
        Draw samples from the composite prior.

        Each component delegates to its own distribution's ``sample()``
        method.  Gaussian parameters use
        :class:`~mach3sbitools.simulator.priors.truncated_gaussian_distribution.TruncatedGaussianDistribution`
        which draws exact, rejection-free samples via the inverse-CDF method.

        :param sample_shape: Batch shape. Pass ``torch.Size([n])`` for
            *n* independent samples.
        :returns: Tensor of shape ``(*sample_shape, n_params)``.
        """
        sample_shape = torch.Size(sample_shape)
        samples = torch.empty(
            (*sample_shape, self.n_params),
            dtype=torch.double,
            device=self.device_handler.device,
        )

        for mask_map in self._priors:
            samples[..., mask_map.mask] = mask_map.distribution.sample(sample_shape).to(
                torch.double
            )

        return samples

    def rsample(self, sample_shape=torch.Size([])) -> torch.Tensor:
        """
        Draw reparameterised samples (where supported by sub-distributions).

        :param sample_shape: Batch shape.
        :returns: Tensor of shape ``(*sample_shape, n_params)``.
        """
        sample_shape = torch.Size(sample_shape)
        samples = torch.empty(*sample_shape, self.n_params, dtype=torch.double)
        for mask_map in self._priors:
            samples[..., mask_map.mask] = mask_map.distribution.rsample(
                sample_shape
            ).to(torch.double)
        return samples

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        log_prob = torch.zeros(
            value.shape[:-1], dtype=torch.double, device=self.device_handler.device
        )
        for mask_map in self._priors:
            lp = mask_map.distribution.log_prob(value[..., mask_map.mask])
            # Some distributions (Uniform, FlippedUniform) return per-parameter
            # log probs of shape (..., n_params) — sum over parameter dimension.
            # Others (MultivariateNormal) already return per-sample scalars (...,).
            if lp.shape != log_prob.shape:
                lp = lp.sum(dim=-1)
            log_prob += lp
        return log_prob

    def check_bounds(self, params: torch.Tensor) -> torch.Tensor:
        lb = self.effective_lower_bounds.to(params.device)
        ub = self.effective_upper_bounds.to(params.device)

        # Coarse check: within [-upper, +upper] for all params
        in_bounds = (params >= lb) & (params <= ub)

        # For flipped params, also exclude the gap region (-lower, +lower)
        if self._flipped_mask.any():
            flipped_lb = self.prior_data.lower_bounds.to(params.device)
            in_gap = params.abs() < flipped_lb
            in_bounds[:, self._flipped_mask] &= ~in_gap[:, self._flipped_mask]

        return self.device_handler.to_tensor(
            in_bounds.all(dim=-1)
        )  # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, output_path: Path) -> None:
        """
        Pickle the prior to *output_path*.

        :param output_path: Destination file path. Parent directories are
            created automatically.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(self, f)

    def to(self, device: torch.device | str) -> "Prior":
        """
        Move all tensors to *device* in-place.

        :param device: Target PyTorch device.
        :returns: ``self``, for chaining.
        """
        self._prior_data = self._prior_data.to(device)
        for i, mask_map in enumerate(self._priors):
            self._priors[i] = mask_map.to(device)
        self._nuisance_filter = self._nuisance_filter.to(device)
        self._flipped_mask = self._flipped_mask.to(device)
        return self


# ── Module-level helpers ───────────────────────────────────────────────────────


def _check_boundary(
    nominal: torch.Tensor,
    error: torch.Tensor,
    lower_bound: torch.Tensor,
    upper_bound: torch.Tensor,
    parameter_names: np.ndarray,
) -> None:
    """
    Warn if any parameter has bounds further than 10σ from its nominal.

    :param nominal: Nominal values, shape ``(n_params,)``.
    :param error: 1σ errors, shape ``(n_params,)``.
    :param lower_bound: Lower bounds, shape ``(n_params,)``.
    :param upper_bound: Upper bounds, shape ``(n_params,)``.
    :param parameter_names: Parameter name strings, shape ``(n_params,)``.
    """
    warning_thresh = 10
    warning_ub = nominal + error * warning_thresh
    warning_lb = nominal - error * warning_thresh

    mask = (lower_bound < warning_lb) | (upper_bound > warning_ub)
    if not any(mask):
        return

    logger.warning(
        f"The following parameters have boundaries > {warning_thresh:d}σ from their prior nominal"
    )
    for param_info in zip(
        parameter_names[mask.cpu().numpy()],
        nominal[mask],
        error[mask],
        lower_bound[mask],
        upper_bound[mask],
    ):
        logger.warning(
            "   '{:s}' | Nominal: {:4f}, Error {:4f} | Lower Bnd {:4f}, Upper Bnd {:4f}".format(
                *param_info
            )
        )


def create_prior(
    simulator_instance: SimulatorProtocol,
    nuisance_pars: list[str] | None = None,
    cyclical_pars: list[str] | None = None,
    flipped_pars: list[str] | None = None,
) -> Prior:
    """
    Convenience function to build a :class:`Prior` from a simulator instance.

    Reads all parameter metadata from *simulator_instance* and constructs the
    appropriate composite prior. Warns about parameters with unusually wide
    bounds (>10σ).

    .. code-block:: console

        prior = create_prior(
            simulator,
            nuisance_pars=["syst_*"],
            cyclical_pars=["angle"],
            flipped_pars=["delta_cp"],
        )
        prior.save(Path("prior.pkl"))

    :param simulator_instance: An object implementing :class:`SimulatorProtocol`.
    :param nuisance_pars: fnmatch patterns for parameters to exclude from the
        prior (e.g. ``['syst_*']``).
    :param cyclical_pars: fnmatch patterns for parameters that should use a
        cyclical sinusoidal prior over ``[-2π, 2π]``.
    :param flipped_pars: fnmatch patterns for parameters that should use a
        bimodal uniform prior over ``[lower, upper] + [-upper, -lower]``,
        where ``lower``/``upper`` come from the simulator's parameter bounds.
    :returns: Configured :class:`Prior` ready for use with ``sbi``.
    """
    logger.info("Creating Prior")
    dh = TorchDeviceHandler()

    nominals = dh.to_tensor(simulator_instance.get_parameter_nominals())
    errors = dh.to_tensor(simulator_instance.get_parameter_errors())
    lower_arr, upper_arr = simulator_instance.get_parameter_bounds()
    lower = dh.to_tensor(lower_arr)
    upper = dh.to_tensor(upper_arr)
    names = np.array(simulator_instance.get_parameter_names(), dtype=str)

    _check_boundary(nominals, errors, lower, upper, names)

    covariance = dh.to_tensor(simulator_instance.get_covariance_matrix())
    flat_pars = [simulator_instance.get_is_flat(i) for i in range(len(names))]

    data = PriorData(
        parameter_names=names,
        nominals=nominals,
        lower_bounds=lower,
        upper_bounds=upper,
        covariance_matrix=covariance,
    )

    prior = Prior(
        prior_data=data,
        flat_msk=flat_pars,
        nuisance_parameters=nuisance_pars,
        cyclical_parameters=cyclical_pars,
        flipped_parameters=flipped_pars,
    )

    get_logger().info("Prior constructed")
    return prior


def load_prior(prior_path: Path, device=torch.device("cpu")) -> Prior:
    """
    Load a pickled :class:`Prior` from disk.

    .. code-block:: console

        prior = load_prior(Path("prior.pkl"))

    :param prior_path: Path to a ``.pkl`` file produced by :meth:`Prior.save`.
    :param device: Device to move the prior to after loading. Defaults to CPU.
    :returns: The loaded :class:`Prior`.
    :raises PriorNotFound: If *prior_path* does not exist or does not contain
        a valid :class:`Prior`.
    """
    if not isinstance(prior_path, Path):
        prior_path = Path(prior_path)

    if not prior_path.is_file() or not prior_path.exists():
        raise PriorNotFound("Could not find prior %s", prior_path)

    with prior_path.open("rb") as f:
        prior = pickle.load(f)

    if not isinstance(prior, Prior):
        raise PriorNotFound(
            "No valid prior in %s. Instead found %s", prior_path, type(prior)
        )

    return prior.to(device)
