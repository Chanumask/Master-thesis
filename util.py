import csv
import os

import yaml
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class Result:
    graph_edges: list

    optimal_value: float
    distribution: Dict[str, float]  # Mapping of states to their probabilities
    solution: Tuple[str, float]  # Most probable state and its probability
    # entanglements: int
    target_value: Optional[float] = None  # Default to None if not provided
    energy_difference: Optional[float] = field(default=None, init=False)
    entanglement_metrics: Dict[str, float] = None

    def compute_energy_difference(self):
        """Calculate the energy difference if target_value is provided."""
        if self.target_value is not None:
            self.energy_difference = abs(self.optimal_value - self.target_value)
        else:
            self.energy_difference = None


def load_config(file_path):
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def log_vqe_results(result, config, file_path):
    """Log the VQE results together with parameters from the config file to a CSV file."""

    # Add all relevant information from the config file.
    relevant_config = {
        **config.get('max_cut', {}),
        **config.get('vqe_parameters', {})
    }
    result_dict = {**relevant_config}  # Start with config fields
    result_dict.update(asdict(result))
    fieldnames = result_dict.keys()

    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(result_dict)
