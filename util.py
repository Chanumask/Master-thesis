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


def setup_results_directory(base_dir="results/"):
    """
    Create a results directory for the current run and return its path.
    Returns the path to the directory for the current run.
    """
    os.makedirs(base_dir, exist_ok=True)

    # Determine the next available Run_index folder
    existing_runs = [
        int(folder.replace("Run_", "")) for folder in os.listdir(base_dir)
        if folder.startswith("Run_") and folder.replace("Run_", "").isdigit()
    ]
    next_index = max(existing_runs) + 1 if existing_runs else 1
    run_folder = os.path.join(base_dir, f"Run_{next_index}")
    os.makedirs(run_folder, exist_ok=True)

    return run_folder


def create_experiment_folder(base_dir="results"):
    """
    Create a new experiment folder in the result directory.
    Returns the path of the created experiment folder.
    """
    # Determine the next available Experiment_<index> folder
    existing_experiments = [int(folder.replace("Experiment_", "")) for folder in os.listdir(base_dir)
                            if folder.startswith("Experiment_") and folder.replace("Experiment_", "").isdigit()]
    next_experiment_index = max(existing_experiments) + 1 if existing_experiments else 1
    experiment_folder = os.path.join(base_dir, f"Experiment_{next_experiment_index}")
    os.makedirs(experiment_folder, exist_ok=True)
    return experiment_folder


def create_run_folder(experiment_folder):
    """
    Create a new run folder inside the given experiment folder.
    Returns the path of the created run folder.
    """
    existing_runs = [int(folder.replace("Run_", "")) for folder in os.listdir(experiment_folder)
                     if folder.startswith("Run_") and folder.replace("Run_", "").isdigit()]
    next_run_index = max(existing_runs) + 1 if existing_runs else 1
    run_folder = os.path.join(experiment_folder, f"Run_{next_run_index}")
    os.makedirs(run_folder, exist_ok=True)
    return run_folder


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


def generate_qpic_circuit(circuit, scale_factor=1.0):
    """
    Generate a qpic-compatible string from a Qiskit QuantumCircuit object.

    Args:
        circuit (QuantumCircuit): A Qiskit quantum circuit.
        scale_factor(float): scaling factor of the qpic elements, also given to latex as fontsize

    Returns:
        str: A string representation of the circuit in qpic format.
    """
    qpic_lines = []
    qpic_lines.append(f"SCALE {scale_factor}")
    num_qubits = circuit.num_qubits
    num_clbits = circuit.num_clbits

    # Add qubit and classical bit declarations
    for i in range(num_qubits):
        qpic_lines.append(f"q{i} W |q_{{{i}}}\\rangle")
    for i in range(num_clbits):
        qpic_lines.append(f"c{i} N")

    measurements = {}
    last_gate_type = None
    gate_counter = 0

    def close_block():
        """Close the current block with LE if it's open."""
        if last_gate_type is not None:
            qpic_lines.append("LE")

    def start_block(gate_type):
        """Start a new block with LB if needed."""
        qpic_lines.append("LB")

    # Iterate through the circuit data to add gates
    for instruction, qubits, clbits in circuit.data:
        gate_name = instruction.name.lower()
        qubit_indices = [q._index for q in qubits]

        # Determine current gate type
        current_gate_type = "rotation" if gate_name in ["rx", "ry", "rz", "x", "h"] else "entanglement"

        # Check for measurement operations
        if gate_name == "measure":
            qubit_index = qubit_indices[0]
            classical_bit_index = clbits[0].index
            measurements[qubit_index] = classical_bit_index
            qpic_lines.append(f"q{qubit_index} M c{classical_bit_index}")

        # Handle gate blocks based on type
        else:
            # Check if we need to close the previous block
            if current_gate_type != last_gate_type:
                if last_gate_type is not None:
                    close_block()
                start_block(current_gate_type)
                last_gate_type = current_gate_type

            # Add gates to the block
            if gate_name == "h":
                qpic_lines.append(f"q{qubit_indices[0]} H")
            elif gate_name == "x":
                qpic_lines.append(f"q{qubit_indices[0]} X")
            elif gate_name == "cx":
                control, target = qubit_indices
                qpic_lines.append(f"q{target} X q{control}")
            elif gate_name == "cz":
                control, target = qubit_indices
                qpic_lines.append(f"q{target} Z q{control}")
            elif gate_name in ["rx", "ry", "rz"]:
                theta = instruction.params[0]
                indeces = f"{str(theta)[2]}{str(theta)[4]}"
                label = f"{gate_name.upper()}(\\theta{indeces})"
                qpic_lines.append(f"q{qubit_indices[0]} G {label} width=40")
            elif gate_name in ["crx", "crz"]:
                control, target = qubit_indices
                theta = instruction.params[0]
                indeces = f"{str(theta)[2]}{str(theta)[4]}"
                label = f"{gate_name.upper()}(\\theta{indeces})"
                qpic_lines.append(f"q{target} G {label} q{control} width=50")
            else:
                qpic_lines.append(f"q{qubit_indices[0]} {gate_name.upper()}")

    # Close any open block at the end
    close_block()

    return '\n'.join(qpic_lines)
