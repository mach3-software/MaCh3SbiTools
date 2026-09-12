from dataclasses import dataclass

import numpy as np

from mach3sbitools.utils import get_logger


@dataclass
class Systematic:
    name: str
    error: np.float64
    nominal: np.float64
    bounds: tuple[np.float64, np.float64]
    correlations: dict[str, np.float64]
    flat_prior: bool
    fixed: bool


@dataclass
class ProcessedSystematics:
    names: np.ndarray
    errors: np.ndarray
    nominals: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    flat_priors: np.ndarray
    fixed: np.ndarray
    covariance: np.ndarray

    def __getitem__(self, mask):
        """
        Select a subset of parameters.

        :param mask: Boolean or index mask over the parameter axis.
        :returns: A new :class:`ProcessedSystematics` holding only those
            parameters, with the covariance sliced on both axes.
        """
        return ProcessedSystematics(
            names=self.names[mask],
            errors=self.errors[mask],
            nominals=self.nominals[mask],
            lower_bounds=self.lower_bounds[mask],
            upper_bounds=self.upper_bounds[mask],
            flat_priors=self.flat_priors[mask],
            fixed=self.fixed[mask],
            covariance=self.covariance[mask][:, mask],
        )

    def __len__(self):
        """
        :returns: Number of parameters held.
        """
        return len(self.names)


def get_corrected_covariance(parameter_handler):
    """
    Read the prior covariance, overriding the known-bad ``delm2_12`` entry.

    :param parameter_handler: MaCh3 parameter handler to read from.
    :returns: The corrected covariance matrix.
    """
    covariance = parameter_handler.get_prior_cov()
    logger = get_logger()
    for i in range(parameter_handler.get_n_pars()):
        par_name = parameter_handler.get_fancy_par_name(i)
        if par_name == "delm2_12":
            logger.info("Correcting covariance for %s to 0.0000018", par_name)
            covariance[i, i] = 0.0000018**2
    return covariance


def process_parameters(parameter_handler) -> ProcessedSystematics:
    """
    Process a list of MaCh3 parameter handler YAML files.

    :param parameter_handler: MaCh3 parameter handler to read from.
    :returns: The parameter properties, packed into a dataclass.
    """
    n_systs = parameter_handler.get_n_pars()
    idx = range(n_systs)

    return ProcessedSystematics(
        names=np.array(
            [parameter_handler.get_fancy_par_name(i) for i in idx], dtype=object
        ),
        errors=np.fromiter(
            (
                parameter_handler.get_par_error(i)
                if parameter_handler.get_fancy_par_name(i) != "delm2_12"
                else 0.0000018
                for i in idx
            ),
            dtype=np.float64,
            count=n_systs,
        ),
        nominals=np.fromiter(
            (parameter_handler.get_par_init(i) for i in idx),
            dtype=np.float64,
            count=n_systs,
        ),
        lower_bounds=np.fromiter(
            (parameter_handler.get_lower_bound(i) for i in idx),
            dtype=np.float64,
            count=n_systs,
        ),
        upper_bounds=np.fromiter(
            (parameter_handler.get_upper_bound(i) for i in idx),
            dtype=np.float64,
            count=n_systs,
        ),
        flat_priors=np.fromiter(
            (parameter_handler.get_flat_prior(i) for i in idx),
            dtype=bool,
            count=n_systs,
        ),
        fixed=np.fromiter(
            (parameter_handler.get_par_fixed(i) for i in idx), dtype=bool, count=n_systs
        ),
        covariance=get_corrected_covariance(parameter_handler),
    )
