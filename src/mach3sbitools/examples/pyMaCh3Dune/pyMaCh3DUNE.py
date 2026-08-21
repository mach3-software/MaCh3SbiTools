from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from yaml import safe_load

from .helpers import process_parameters

try:
    # NOTE: assuming `Manager` is exposed from the core pyMaCh3 module as
    # `pyMaCh3.manager.Manager` (not from pyMaCh3_DUNE). Adjust this import
    # to match wherever your core bindings actually expose it.
    from pyMaCh3.manager import Manager
    from pyMaCh3_DUNE import parameters, samples

    HAS_PYMACH3 = True
except ImportError:
    HAS_PYMACH3 = False

if TYPE_CHECKING:
    from pyMaCh3.manager import Manager
    from pyMaCh3_DUNE import parameters, samples

from mach3sbitools.utils.logger import get_logger

logger = get_logger()


class pyMaCh3DUNESimulator:
    def __init__(self, fitter_config: Path):
        """
        PyMaCh3 simulator object, a helper function around the MaCh3 simulator implementation
        :param fitter_config: MaCh3 configuration file
        """
        if not HAS_PYMACH3:
            raise ImportError(
                "Trying to instantiate pyMaCh3DUNESimulator without a pyMaCh3_DUNE install!"
            )

        # Read in MaCh3 Config
        fitter_config = Path(fitter_config)
        if not fitter_config.is_file():
            raise FileNotFoundError(fitter_config)

        with open(fitter_config) as f:
            yaml_cfg = safe_load(f)

        # We assume a single parameter config is used
        systematics_opts = yaml_cfg.get("General", {}).get("Systematics", {})

        systematic_configs = systematics_opts.get("XsecCovFile", [])

        self.parameter_handler = parameters.ParameterHandlerGeneric(
            [str(s) for s in systematic_configs]
        )

        # We'll use this in a minute! (contains everything about or systematic model!)
        self.parameter_properties = process_parameters(self.parameter_handler)

        # We also need to get additional fixed pars from the config
        additional_fixed_names = systematics_opts.get("XsecFix", [])
        additional_fixed_mask = np.array(
            [n in additional_fixed_names for n in self.parameter_properties.names]
        )

        # We'll use this a lot
        self._fixed_mask = self.parameter_properties.fixed | additional_fixed_mask
        self.n_params = len(self.parameter_properties[~self._fixed_mask])

        logger.info(f"Fixing {self.parameter_properties[self._fixed_mask]}")

        # Saves doing this every time!
        self._parameter_properties_masked = self.parameter_properties[~self._fixed_mask]

        # Now we load in the samples via the C++ factory, so the sample-type
        # dispatch (BeamFD / BeamND / Atm / BeamNDGAr), oscillation handler
        # setup, and ND covariance loading all live in one place.
        self.samples = self._get_sample_handlers(fitter_config, self.parameter_handler)

        self._data = np.concatenate(
            [
                s.get_data_array(i)
                for s in self.samples
                for i in range(s.get_n_samples())
            ]
        )
        # Another helper
        self.n_bins = len(self._data)

    # -----------------
    # Simulator protocol methods
    # -----------------
    def simulate(self, theta: list[float] | np.ndarray) -> np.ndarray:
        """
        Run the simulation step
        :param theta: Parameter values
        :return: Simulated data
        """
        self._set_parameter_values(theta)
        return np.concatenate(
            [s.get_mc_array(i) for s in self.samples for i in range(s.get_n_samples())]
        )

    def get_parameter_names(self):
        return self._parameter_properties_masked.names

    def get_parameter_bounds(self):
        return (
            self._parameter_properties_masked.lower_bounds,
            self._parameter_properties_masked.upper_bounds,
        )

    def get_is_flat(self, i: int):
        return self._parameter_properties_masked.flat_priors[i]

    def get_data_bins(self):
        return self._data

    def get_parameter_nominals(self):
        return self._parameter_properties_masked.nominals

    def get_parameter_errors(self):
        return self._parameter_properties_masked.errors

    def get_covariance_matrix(self):
        return self._parameter_properties_masked.covariance

    def get_log_likelihood(self, theta: list[float] | np.ndarray) -> float:
        lower, upper = self.get_parameter_bounds()
        if np.any(theta < lower) or np.any(theta > upper):
            return float(-np.inf)
        try:
            self._set_parameter_values(theta)
            prior_llh: float = self.parameter_handler.calculate_likelihood()
            if prior_llh > 1234567:
                return float(-np.inf)

            sample_llh: float = np.sum([s.get_likelihood() for s in self.samples])

            return -sample_llh - prior_llh
        except RuntimeError:
            return float(-np.inf)

    # -----------------
    # Helpers
    # -----------------

    @classmethod
    def _get_sample_handlers(
        cls,
        fitter_config: Path,
        parameter_handler: parameters.ParameterHandlerGeneric,
    ):
        """
        Load in the samples from the MaCh3 fitter config via the C++
        MaCh3DuneSampleFactory, which reads General:DUNESamples itself,
        dispatches on each sample's SampleHandlerName (BeamFD / BeamND /
        Atm / BeamNDGAr), and sets up any shared oscillation handlers and
        ND covariance objects needed along the way.

        :param fitter_config: path to the MaCh3 fitter config (same one
            used to build `parameter_handler`)
        :param parameter_handler: parameter handler to associate with the samples
        :return: list of constructed SampleHandlerBase instances
        """
        fit_manager = Manager(str(fitter_config))
        return samples.MaCh3DuneSampleFactory(fit_manager, parameter_handler)

    def _set_parameter_values(self, theta: list[float] | np.ndarray):
        """
        Set the parameter values to some value and reweight
        :param theta:
        :return:
        """
        # Cast to nd array
        if isinstance(theta, list):
            theta = np.array(theta)

        if len(theta) != self.n_params:
            raise ValueError(f"theta must have {self.n_params} elements")

        # Make a copy (little expensive)
        set_values = self.parameter_properties.nominals.copy()
        # Set non-fixed values
        set_values[~self._fixed_mask] = theta

        self.parameter_handler.set_parameters(set_values)

        for sample in self.samples:
            sample.reweight()
