import csv
import os

import numpy as np
import yaml
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field, asdict


def generate_deterministic_seeds(base_seed, num_seeds):
    """Generate a list of deterministic seeds."""
    rng = np.random.default_rng(base_seed)
    return [rng.integers(0, 10000) for _ in range(num_seeds)]


@dataclass
class Result:
    seed: int
    graph_edges: list

    optimal_value: float
    distribution: Dict[str, float]  # Mapping of states to their probabilities
    solution: Tuple[str, float]  # Most probable state and its probability
    cut_cost: int  # summed weights of edges that have been cut by the most probable solution

    brute_forced_cost: Optional[float] = None
    cut_cost_difference: Optional[float] = field(default=None, init=False)
    approximation_ratio: Optional[float] = field(default=None, init=False)

    entanglement_metrics: Dict[str, float] = None
    energy_history: [float] = None

    def compute_cut_cost_difference(self):
        """Calculate the energy difference if target_value is provided."""
        if self.brute_forced_cost is not None:
            self.cut_cost_difference = abs(self.cut_cost - self.brute_forced_cost)
        else:
            self.cut_cost_difference = None

    def compute_approximation_ratio(self):
        """Calculate the approximation ratio based on the VQE cut cost and brute-forced optimal cost."""
        if self.brute_forced_cost and self.brute_forced_cost != 0:
            self.approximation_ratio = self.cut_cost / self.brute_forced_cost
        else:
            self.approximation_ratio = None


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
