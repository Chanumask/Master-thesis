import csv
import os

import matplotlib.pyplot as plt
import pandas as pd
from qiskit import transpile
from qiskit.primitives import Sampler
from qiskit_aer import AerSimulator


def run_circuit(circuit):
    simulator = AerSimulator()
    simulator.set_options(seed_simulator=1)
    result = simulator.run(transpile(circuit, simulator), shots=1024).result()
    counts = result.get_counts(circuit)
    return counts


def get_solution(optimal_state):
    """Run the sampler, obtain the solution distribution and get the most probable solution."""
    sampler = Sampler(options={"shots": 1024})
    distribution = sampler.run([optimal_state]).result().quasi_dists[0]
    solution = max(distribution.binary_probabilities().items(), key=lambda x: x[1])
    return distribution, solution


def plot_solution(dist, sol, filename):
    """Plots histogram and prints most probable solution."""
    # Check if 'dist' indeed includes all states, if not, the previous method will be necessary
    # Here we assume it includes all states

    # Extracting probability data
    probabilities = dist.binary_probabilities()

    # Plotting
    fig, ax = plt.subplots()
    ax.bar(probabilities.keys(), probabilities.values(), color='blue')
    ax.set_xlabel('State')
    ax.set_ylabel('Probability')
    ax.set_title('Probability Distribution of Quantum States')
    plt.xticks(rotation=90)  # Rotate state labels for better visibility
    plt.tight_layout()

    # Save the figure
    plt.savefig(filename)
    plt.show()

    # Print most probable solution
    print(f"State: {sol[0]}, Probability: {sol[1]}")
    for i, group in enumerate(reversed(sol[0])):
        print(f"Node {i}: {group}")


def evaluate_vqe_performance(optimal_value, target_value=None):
    """Evaluate and print the performance of VQE."""
    print(f"Optimal Energy: {optimal_value:.4f}")
    if target_value is not None:
        print(f"Target Energy: {target_value:.4f}")
        print(f"Energy Difference: {abs(optimal_value - target_value):.4f}")


def log_vqe_results(nodes, connectivity, reps, optimal_value, entangle_ratio, entanglements,
                    file_path, target_value=None):
    """Log the VQE results to a CSV file."""
    fieldnames = ['nodes', 'connectivity', 'ansatz_reps', 'optimal_value', 'target_value', 'energy_difference',
                  'entangle_ratio', 'n_entanglements']
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # Write header only if the file does not exist

        # Calculate energy difference if target value is provided
        energy_difference = abs(optimal_value - target_value) if target_value is not None else 'N/A'

        writer.writerow({
            'nodes': nodes,
            'connectivity': connectivity,
            'ansatz_reps': reps,
            'optimal_value': optimal_value,
            'target_value': target_value if target_value is not None else 'N/A',
            'energy_difference': energy_difference,
            'entangle_ratio': entangle_ratio,
            'n_entanglements': entanglements,
        })


def evaluate_csv(filepath):
    data = pd.read_csv(filepath)

    average_optimal_values = data.groupby('n_entanglements')['optimal_value'].mean()
    sample_counts = data['n_entanglements'].value_counts().sort_index()

    print("Average Optimal Value per Number of Entanglements:")
    print(average_optimal_values)
    print("\nSample Count per Number of Entanglements:")
    print(sample_counts)
