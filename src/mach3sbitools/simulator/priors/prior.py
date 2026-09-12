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
from typing import TypeAlias

import numpy as np
import torch
from torch.distributions import Uniform, constraints

from mach3sbitools.utils import get_device, get_logger, to_tensor

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
    """

    mask: torch.Tensor
    distribution: torch.distributions.Distribution

    def to(self, device: torch.device | str) -> "MaskDistributionMap":
        """
        Move *mask* and *distribution* to *device*.

        Custom distributions (Cyclical/FlippedUniform/TruncatedGaussian)
        implement their own ``to()``. ``torch.distributions.Uniform`` (used
        for flat priors) does NOT implement ``to()`` — it's a plain
        Distribution, not an nn.Module — so it must be rebuilt explicitly
        or it silently stays on its original device.

        :param device: Target torch device.
        :returns: A map whose mask and distribution live on *device*.
        """
        dist = self.distribution

        if hasattr(dist, "to") and callable(getattr(dist, "to")):
            dist = dist.to(device)
        elif isinstance(dist, Uniform):
            dist = Uniform(dist.low.to(device), dist.high.to(device))
        else:
            raise TypeError(
                f"Don't know how to move distribution of type "
                f"{type(dist).__name__} to device {device}. "
                f"Add a .to() method or handle it explicitly here."
            )

        return MaskDistributionMap(mask=self.mask.to(device), distribution=dist)


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

    #: Filtered prior data, rebuilt lazily and dropped by :meth:`to`.
    _prior_data_cache: PriorData | None
    #: Cached ``(lower, upper)`` support bounds, dropped by :meth:`to`.
    _effective_bounds: tuple[torch.Tensor, torch.Tensor] | None

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
        self._device = get_device()
        self._prior_data = prior_data.to(self._device)
        self._invalidate_cache()

        # Apply nuisance filter once — masks are built against the filtered set
        # and cannot be safely remapped if the filter changes afterwards.
        self.nuisance_filter = self._build_nuisance_filter(nuisance_parameters)
        self._priors: list[MaskDistributionMap] = []

        n_params = len(self.prior_data.nominals)

        # ── Cyclical mask ──────────────────────────────────────────────────
        if cyclical_parameters:
            cyclical_mask_ = [
                any(fnmatch.fnmatch(p, c) for c in cyclical_parameters)
                for p in self.prior_data.parameter_names
            ]
            cyclical_mask = to_tensor(cyclical_mask_)
        else:
            cyclical_mask = torch.zeros(n_params, dtype=torch.bool, device=self._device)

        if any(cyclical_mask):
            # Mutate _prior_data directly, not a temporary slice
            full_cyclical_mask = torch.zeros(
                len(self._prior_data.parameter_names),
                dtype=torch.bool,
                device=self._device,
            )
            full_cyclical_mask[self.nuisance_filter] = cyclical_mask
            self._prior_data.lower_bounds[full_cyclical_mask] = -2 * torch.pi
            self._prior_data.upper_bounds[full_cyclical_mask] = 2 * torch.pi
            self._priors.append(self._get_cyclical_map(cyclical_mask))

        self.cyclical_mask = cyclical_mask

        # ── Flipped-uniform mask ───────────────────────────────────────────
        if flipped_parameters:
            flipped_mask_ = [
                any(fnmatch.fnmatch(p, f) for f in flipped_parameters)
                for p in self.prior_data.parameter_names
            ]
            flipped_mask = to_tensor(flipped_mask_).bool()
            # Flipped params must not also be cyclical
            flipped_mask = flipped_mask & ~cyclical_mask
        else:
            flipped_mask = torch.zeros(n_params, dtype=torch.bool, device=self._device)

        # Store for use in check_bounds — flipped params are valid in EITHER
        # region so the standard lower/upper bounds check would incorrectly
        # reject samples drawn from the negative region.
        self._flipped_mask = flipped_mask

        if any(flipped_mask):
            self._priors.extend(self._get_flipped_maps(flipped_mask))

        # ── Flat mask ──────────────────────────────────────────────────────
        # Guard against None so the tensor conversion doesn't crash.
        flat_msk = flat_msk if flat_msk is not None else [False] * n_params

        flat_msk_tensor = to_tensor(flat_msk).bool()
        flat_msk_filtered = flat_msk_tensor[self.nuisance_filter]

        flat_mask = flat_msk_filtered & ~cyclical_mask & ~flipped_mask
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
            return torch.ones(n_pars, dtype=torch.bool, device=self._device)

        keep = [
            not any(
                (p == n) if "*" not in n and "?" not in n else fnmatch.fnmatch(p, n)
                for n in nuisance_patterns
            )
            for p in self._prior_data.parameter_names
        ]

        return to_tensor(keep)

    # ── Private distribution builders ──────────────────────────────────────────

    def _get_cyclical_map(self, cyclical_mask: torch.Tensor) -> MaskDistributionMap:
        """
        Build the cyclical sub-prior.

        :param cyclical_mask: Mask selecting cyclical parameters.
        :returns: Mask/distribution pair for those parameters.
        """
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
                len(flipped_mask), dtype=torch.bool, device=self._device
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
        """
        Build the uniform sub-prior for parameters flagged flat.

        :param flat_mask: Mask selecting flat parameters.
        :returns: Mask/distribution pair for those parameters.
        """
        flat_data = self.prior_data[flat_mask]
        flat_dist = Uniform(flat_data.lower_bounds, flat_data.upper_bounds)
        return MaskDistributionMap(flat_mask, flat_dist)

    def _get_gaussian_map(self, gaussian_mask: torch.Tensor) -> MaskDistributionMap:
        """
        Build the truncated-Gaussian sub-prior for the remaining parameters.

        :param gaussian_mask: Mask selecting Gaussian-constrained parameters.
        :returns: Mask/distribution pair for those parameters.
        """
        gaussian_data = self.prior_data[gaussian_mask]
        dist = TruncatedGaussianDistribution(
            mean=gaussian_data.nominals,
            covariance=gaussian_data.covariance_matrix,
            lower_bounds=gaussian_data.lower_bounds,
            upper_bounds=gaussian_data.upper_bounds,
        )
        return MaskDistributionMap(gaussian_mask, dist)

    # ── Properties ─────────────────────────────────────────────────────────────
    def _compute_effective_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build the bounds pair used for support checks.

        A flipped parameter is valid on ``[-upper, -lower]`` as well as
        ``[lower, upper]``, so its effective lower bound is ``-upper``.

        :returns: Tuple of ``(lower, upper)`` bounds, shape ``(n_params,)``.
        """
        lower = self.prior_data.lower_bounds.clone()
        if self._flipped_mask.any():
            lower[self._flipped_mask] = -self.prior_data.upper_bounds[
                self._flipped_mask
            ]
        return lower, self.prior_data.upper_bounds

    @property
    def effective_lower_bounds(self) -> torch.Tensor:
        """
        :returns: Lower bounds of shape ``(n_params,)``, widened for flipped
            parameters.
        """
        if self._effective_bounds is None:
            self._effective_bounds = self._compute_effective_bounds()
        return self._effective_bounds[0]

    @property
    def effective_upper_bounds(self) -> torch.Tensor:
        """
        :returns: Upper bounds of shape ``(n_params,)``.
        """
        if self._effective_bounds is None:
            self._effective_bounds = self._compute_effective_bounds()
        return self._effective_bounds[1]

    @property
    def device(self) -> torch.device:
        """
        :returns: The device every tensor in this prior lives on.
        """
        return self._device

    @property
    def prior_data(self) -> PriorData:
        """
        Active :class:`PriorData` after applying the nuisance filter.

        Filtering rebuilds four tensors including an O(n²) slice of the
        covariance matrix, and this is read on every bounds check, so the
        result is cached. The nuisance filter is fixed at construction, so
        only :meth:`to` can invalidate it.

        :returns: The filtered prior data.
        """
        if self._prior_data_cache is None:
            self._prior_data_cache = self._prior_data[self.nuisance_filter]
        return self._prior_data_cache

    def _invalidate_cache(self) -> None:
        """Drop the derived tensors cached off :attr:`_prior_data`."""
        self._prior_data_cache = None
        self._effective_bounds = None

    @property
    def mean(self) -> torch.Tensor:
        """
        :returns: Prior mean — the nominal parameter values.
        """
        return to_tensor(self.prior_data.nominals)

    @property
    def n_params(self) -> int:
        """
        :returns: Number of active (non-nuisance) parameters.
        """
        return len(self.prior_data.nominals)

    @property
    def variance(self) -> torch.Tensor:
        """
        Per-parameter prior variance, assembled from all sub-distributions.

        :returns: Variance tensor of shape ``(n_params,)``.

        The mask for each sub-distribution is sized against the filtered
        parameter set (same as the tensor being filled), so shapes are always
        consistent.
        """
        variance = torch.zeros(self.n_params, device=self._device)
        for mask_map in self._priors:
            variance[mask_map.mask] = mask_map.distribution.variance
        return variance

    @property
    def support(self) -> constraints.Constraint:
        """
        :returns: The interval constraint sbi uses to reject out-of-bounds
            proposals, built from the effective bounds.
        """
        return constraints.independent(
            constraints.interval(
                self.effective_lower_bounds,
                self.effective_upper_bounds,
            ),
            1,
        )

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
            device=self._device,
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
        """
        Evaluate the joint prior log-density.

        Each sub-distribution contributes the parameters its mask selects;
        the contributions are summed as the blocks are independent.

        :param value: Parameters of shape ``(..., n_params)``.
        :returns: Log-density of shape ``(...,)``.
        """
        log_prob = torch.zeros(
            value.shape[:-1], dtype=torch.double, device=self._device
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
        """
        Test whether each row lies inside the prior support.

        Flipped parameters are handled specially: the forbidden gap
        ``(-lower, lower)`` around zero is excluded as well as the outer
        bounds.

        :param params: Parameters of shape ``(..., n_params)``.
        :returns: Boolean mask of shape ``(...,)``.
        """
        lb = self.effective_lower_bounds.to(params.device)
        ub = self.effective_upper_bounds.to(params.device)

        # Coarse check: within [-upper, +upper] for all params
        in_bounds = (params >= lb) & (params <= ub)

        # For flipped params, also exclude the gap region (-lower, +lower)
        if self._flipped_mask.any():
            flipped_lb = self.prior_data.lower_bounds.to(params.device)
            in_gap = params.abs() < flipped_lb
            in_bounds[..., self._flipped_mask] &= ~in_gap[..., self._flipped_mask]

        # Stay on the caller's device: this runs inside the rejection-sampling
        # loop, where a hop to another device would cost a sync per batch.
        return in_bounds.all(dim=-1)

    # ── Persistence ────────────────────────────────────────────────────────────
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
        self._device = torch.device(device)
        self._prior_data = self._prior_data.to(self._device)
        for i, mask_map in enumerate(self._priors):
            self._priors[i] = mask_map.to(self._device)
        self.nuisance_filter = self.nuisance_filter.to(self._device)
        self._flipped_mask = self._flipped_mask.to(self._device)
        self.cyclical_mask = self.cyclical_mask.to(self._device)
        self._invalidate_cache()

        return self

    # ── Pickling ───────────────────────────────────────────────────────────────
    def __getstate__(self) -> dict:
        """
        Drop derived state so it cannot be restored stale.

        The cached tensors and the recorded device describe the machine that
        pickled the prior, not the one that will load it.

        :returns: The picklable instance dictionary.
        """
        state = self.__dict__.copy()
        for derived in ("_prior_data_cache", "_effective_bounds", "_device"):
            state.pop(derived, None)
        return state

    def __setstate__(self, state: dict) -> None:
        """
        Restore a pickled prior onto *this* machine's device.

        A prior pickled on a GPU node and loaded on a CPU one would otherwise
        keep the device it was saved with. Device detection is redone here and
        every tensor moved to match, so the prior is always self-consistent
        after a load.

        :param state: The unpickled instance dictionary.
        """
        # Legacy pickles carry a TorchDeviceHandler instance; it is replaced
        # by the detection below.
        state.pop("device_handler", None)
        self.__dict__.update(state)
        self._invalidate_cache()
        self.to(get_device())


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

    nominals = to_tensor(simulator_instance.get_parameter_nominals())
    errors = to_tensor(simulator_instance.get_parameter_errors())
    lower_arr, upper_arr = simulator_instance.get_parameter_bounds()
    lower = to_tensor(lower_arr)
    upper = to_tensor(upper_arr)
    names = np.array(simulator_instance.get_parameter_names(), dtype=str)

    _check_boundary(nominals, errors, lower, upper, names)

    covariance = to_tensor(simulator_instance.get_covariance_matrix())
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


def load_prior(prior_path: Path, device: torch.device | str | None = None) -> Prior:
    """
    Load a pickled :class:`Prior` from disk.

    .. code-block:: console

        prior = load_prior(Path("prior.pkl"))

    :param prior_path: Path to a ``.pkl`` file produced by :meth:`Prior.save`.
    :param device: Device to move the prior to after loading. Defaults to
        whatever :func:`~mach3sbitools.utils.get_device` selects.
    :returns: The loaded :class:`Prior`.
    :raises PriorNotFound: If *prior_path* does not exist or does not contain
        a valid :class:`Prior`.
    """
    if not isinstance(prior_path, Path):
        prior_path = Path(prior_path)

    if not prior_path.is_file():
        raise PriorNotFound(f"Could not find prior {prior_path}")

    with prior_path.open("rb") as f:
        prior = pickle.load(f)

    if not isinstance(prior, Prior):
        raise PriorNotFound(
            f"No valid prior in {prior_path}. Instead found {type(prior)}"
        )

    return prior.to(device if device is not None else get_device())
