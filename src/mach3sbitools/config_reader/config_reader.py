'''
Reads yaml config to setup files
'''

import yaml
from pathlib import Path
from pydantic import validate_call
from dataclasses import dataclass, field
from typing import Literal, Optional

from .config import TrainingConfig, PosteriorConfig

# Config dataclass groups
@validate_call
@dataclass
class LoggerOptions:
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    
@validate_call
@dataclass
class SimulatorOptions:
    output_file: Path
    simulator_module: str
    simulator_class: str
    config: Path
    
@validate_call
@dataclass 
class ParameterOptions:
    nuisance_pars: list[str] = field(default_factory=list)
    cyclical_pars: list[str] = field(default_factory=list)
    flipped_pars: list[str] = field(default_factory=list)

@validate_call
@dataclass
class SamplingOptions:
    posterior: Path
    n_samples: int = 10_000
    oversampling_factor: int = 5
    max_sampling_batch: int = 10_000

@validate_call
@dataclass
class DiagnosticOptions:
    n_prior_samples: int = 200
    n_posterior_samples: int = 1000
    make_sbc_rank: bool = False
    make_expected_coverage: bool = False
    max_tarp: bool = False
    make_logl_comp: bool = False

# Registry
_OPTION_REGISTRY = {
    "Training": TrainingConfig,
    "Posterior": PosteriorConfig,
    "Logger": LoggerOptions,
    "Parameters": ParameterOptions,
    "Simulation": SimulatorOptions,
    "Sampling": SamplingOptions,
    "Diagnostics": DiagnosticOptions,
}

OPTIONS = Literal[*_OPTION_REGISTRY.keys()]

class ConfigReader:
    def __init__(self, config_file: Path):
        """Configuration reader for applications

        :param config_file: Config file path
        :raises FileNotFoundError: Cannot find config
        """
        if not config_file.is_file():
            raise FileNotFoundError(f"Cannot find {config_file}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        self._options = {r: _OPTION_REGISTRY[r](**v) for r, v in config.items()}
        
    def get_opt(self, opt_category: str):
        if opt_category not in self._options:
            raise KeyError(f"Cannot find {self._options} in config")
        
        return self._options[opt_category]