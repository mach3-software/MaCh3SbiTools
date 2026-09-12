from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from yaml import safe_load

from .helpers import process_parameters

try:
    import pyMaCh3_tutorial as m3

    HAS_PYMACH3 = True
except ImportError:
    HAS_PYMACH3 = False

if TYPE_CHECKING:
    import pyMaCh3_tutorial as m3

from mach3sbitools.utils.logger import get_logger

logger = get_logger()


class pyMaCh3Simulator:
    def __init__(self, fitter_config: Path):
        """
        PyMaCh3 simulator object, a helper function around the MaCh3 simulator implementation
        :param fitter_config: MaCh3 configuration file
        """
        if not HAS_PYMACH3:
            raise ImportError(
                "Trying to instantiate pyMaCh3Simulator without a pyMaCh3 install!"
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

        self.parameter_handler = m3.parameters.ParameterHandlerGeneric(
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

        # Now we load in the samples
        self.samples = self._get_sample_handlers(yaml_cfg, self.parameter_handler)

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
        Run the simulation step.

        :param theta: Parameter values.
        :returns: Simulated data bins.
        """
        self._set_parameter_values(theta)
        return np.concatenate(
            [s.get_mc_array(i) for s in self.samples for i in range(s.get_n_samples())]
        )

    def get_parameter_names(self):
        """
        :returns: Names of the active parameters.
        """
        return self._parameter_properties_masked.names

    def get_parameter_bounds(self):
        """
        :returns: Tuple of ``(lower_bounds, upper_bounds)`` arrays.
        """
        return (
            self._parameter_properties_masked.lower_bounds,
            self._parameter_properties_masked.upper_bounds,
        )

    def get_is_flat(self, i: int):
        """
        :param i: Parameter index.
        :returns: ``True`` if parameter *i* has a flat prior.
        """
        return self._parameter_properties_masked.flat_priors[i]

    def get_data_bins(self):
        """
        :returns: The observed data bins.
        """
        return self._data

    def get_parameter_nominals(self):
        """
        :returns: Nominal (central) values of the active parameters.
        """
        return self._parameter_properties_masked.nominals

    def get_parameter_errors(self):
        """
        :returns: Prior uncertainties of the active parameters.
        """
        return self._parameter_properties_masked.errors

    def get_covariance_matrix(self):
        """
        :returns: Covariance matrix of the active parameters.
        """
        return self._parameter_properties_masked.covariance

    def get_log_likelihood(self, theta: list[float] | np.ndarray) -> float:
        """
        Evaluate the total log-likelihood at *theta*.

        :param theta: Parameter values.
        :returns: Sample plus prior log-likelihood, or ``-inf`` if the prior
            term signals an out-of-bounds point.
        """
        self._set_parameter_values(theta)
        prior_llh: float = self.parameter_handler.calculate_likelihood()
        if prior_llh > 1234567:
            return float(-np.inf)

        sample_llh: float = np.sum([s.get_likelihood() for s in self.samples])

        return -sample_llh - prior_llh

    # -----------------
    # Helpers
    # -----------------

    @classmethod
    def _get_sample_handlers(
        cls,
        yaml_cfg: dict,
        parameter_handler: m3.parameters.ParameterHandlerGeneric,
    ) -> list[m3.samples.SampleHandlerTutorial]:
        """
        Load in the samples from the MaCh3 fitter config.

        :param yaml_cfg: Main config.
        :param parameter_handler: A list of fitter configs.
        :returns: The configured sample handlers.
        """
        samples = yaml_cfg.get("General", {}).get("TutorialSamples")
        if samples is None:
            raise ValueError("TutorialSamples is required in fitter config")

        return [
            m3.samples.SampleHandlerTutorial(str(s), parameter_handler) for s in samples
        ]

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
