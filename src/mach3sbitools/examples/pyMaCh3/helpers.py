from dataclasses import dataclass

import numpy as np


@dataclass
class Systematic:
    name: str
    error: float
    nominal: float
    bounds: tuple[float, float]
    correlations: dict[str, float]
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
            (parameter_handler.get_par_error(i) for i in idx),
            dtype=float,
            count=n_systs,
        ),
        nominals=np.fromiter(
            (parameter_handler.get_par_init(i) for i in idx), dtype=float, count=n_systs
        ),
        lower_bounds=np.fromiter(
            (parameter_handler.get_lower_bound(i) for i in idx),
            dtype=float,
            count=n_systs,
        ),
        upper_bounds=np.fromiter(
            (parameter_handler.get_upper_bound(i) for i in idx),
            dtype=float,
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
        covariance=parameter_handler.get_prior_cov(),
    )
