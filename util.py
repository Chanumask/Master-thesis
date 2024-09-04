import csv
import os

import yaml
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field, asdict
import networkx as nx


@dataclass
class Result:
    nodes: int
    graph: nx.classes.Graph
    reps: int
    optimal_value: float
    distribution: Dict[str, float]  # Mapping of states to their probabilities
    solution: Tuple[str, float]  # Most probable state and its probability
    entangle_ratio: float
    # entanglements: int
    target_value: Optional[float] = None  # Default to None if not provided
    energy_difference: Optional[float] = field(default=None, init=False)

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


def log_vqe_results(result, file_path):
    """Log the VQE results to a CSV file."""
    result_dict = asdict(result)

    fieldnames = result_dict.keys()

    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # Write header only if the file does not exist

        writer.writerow(result_dict)
