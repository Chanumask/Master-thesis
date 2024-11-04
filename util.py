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


import re

def generate_qpic_circuit(circuit):
    """
    Generate a qpic-compatible string from a Qiskit QuantumCircuit object.

    Args:
        circuit (QuantumCircuit): A Qiskit quantum circuit.

    Returns:
        str: A string representation of the circuit in qpic format.
    """
    # Initialize the qpic string
    qpic_lines = []

    # Get the number of qubits
    num_qubits = circuit.num_qubits

    # Add qubit declarations (W denotes wire)
    for i in range(num_qubits):
        qpic_lines.append(f"q{i} W |q_{{{i}}}\\rangle")

    # Iterate through the circuit data to add gates
    for instruction, qubits, clbits in circuit.data:
        gate_name = instruction.name.lower()
        qubit_indices = [q._index for q in qubits]

        if gate_name == "h":
            # Hadamard gate
            qpic_lines.append(f"q{qubit_indices[0]} H")
        elif gate_name == "x":
            # Pauli-X gate
            qpic_lines.append(f"q{qubit_indices[0]} X")
        elif gate_name == "cx":
            # CNOT gate (control and target)
            qpic_lines.append(f"q{qubit_indices[0]} q{qubit_indices[1]} CNOT")
        elif gate_name == "cz":
            # Controlled-Z gate (control and target)
            qpic_lines.append(f"q{qubit_indices[0]} q{qubit_indices[1]} CZ")
        elif gate_name in ["rx", "ry", "rz"]:
            # Rotation gates with parameter theta
            theta = instruction.params[0]
            # Convert theta to string
            theta_str = str(theta)
            # Replace 'θ' with LaTeX '\theta'
            theta_str = theta_str.replace('θ', '\\theta')
            # Enclose subscripts in braces and replace underscores with commas
            # e.g., '\theta_0_0' -> '\theta_{0,0}'
            match = re.match(r'(\\theta)_(.+)', theta_str)
            if match:
                base = match.group(1)
                subscripts = match.group(2).split('_')
                subscripts_str = ','.join(subscripts)
                theta_str = f"{base}_{{{subscripts_str}}}"
            # Ensure parentheses are properly closed
            qpic_lines.append(f"q{qubit_indices[0]} {gate_name.upper()}({theta_str})")
        elif gate_name == "measure":
            # Measurement gate
            qpic_lines.append(f"q{qubit_indices[0]} M")
        else:
            # Other gates
            qpic_lines.append(f"q{qubit_indices[0]} {gate_name.upper()}")

    # Join all lines into a single string
    qpic_str = '\n'.join(qpic_lines)

    return qpic_str

