import csv
import os
import numpy as np

import networkx as nx
from qiskit import QuantumCircuit
from qiskit_optimization.applications import Maxcut
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.minimum_eigensolvers import VQE

from util import get_solution, plot_solution


def generate_random_graph(num_nodes, connectivity_prob, max_weight):
    """Generate a random graph using the Erdős-Rényi model."""
    g = nx.erdos_renyi_graph(n=num_nodes, p=connectivity_prob)
    # print(g)
    # Assign random weights to each edge
    for (u, v) in g.edges():
        g.edges[u, v]['weight'] = np.random.randint(1, max_weight)  # Random weights between 1 and 10

    return g


def construct_weighted_edges_from_graph(graph):
    """Convert a NetworkX graph into weighted edges."""
    weighted_edges = [(u, v, d['weight']) for u, v, d in graph.edges(data=True)]
    # print(f"graph representation: {weighted_edges}")
    return weighted_edges


def construct_graph(weighted_edges):
    """Construct a NetworkX graph from weighted edges."""
    g = nx.Graph()
    g.add_weighted_edges_from(weighted_edges)
    return g


def get_hamiltonian(nxgraph):
    """Generate the Hamiltonian for the MaxCut problem."""
    maxcut = Maxcut(nxgraph)
    quadratic = maxcut.to_quadratic_program()
    return quadratic.to_ising()


def add_cyclic_entanglement(circuit, entangle_prob):
    """Add cyclic entanglement using CX gates at the beginning of the circuit."""
    num_qubits = circuit.num_qubits
    entanglements_added = 0

    for i in range(num_qubits):
        if np.random.rand() < entangle_prob:  # Randomly decide to add entanglement
            circuit.cx(i, (i + 1) % num_qubits)
            entanglements_added += 1

    return circuit, entanglements_added


def create_custom_entanglement_map(num_qubits, remove_entangle_prob):
    """Create a custom entanglement map with some entanglements removed."""
    entanglement_map = []
    for i in range(num_qubits - 1):
        if np.random.rand() >= remove_entangle_prob:
            entanglement_map.append((i, i + 1))
    # Optionally add entanglement between the last and first qubits
    if np.random.rand() >= remove_entangle_prob:
        entanglement_map.append((num_qubits - 1, 0))
    return entanglement_map


def solve_maxcut(weighted_edges, qubits, reps, entangle_probability):
    """Solve the MaxCut problem using VQE and return the solution."""
    graph = construct_graph(weighted_edges)
    hamiltonian, offset = get_hamiltonian(graph)

    # Create a custom entanglement map for the ansatz
    remove_entangle_prob = 1 - entangle_probability
    entanglement_map = create_custom_entanglement_map(qubits, remove_entangle_prob)
    n_entanglements_ansatz = len(entanglement_map)

    ansatz = RealAmplitudes(num_qubits=qubits, reps=reps)  # read paper

    full_circuit = QuantumCircuit(qubits)
    full_circuit, n_entanglements = add_cyclic_entanglement(full_circuit, entangle_probability)
    full_circuit.compose(ansatz, inplace=True)

    # print(full_circuit.decompose().draw())
    # print(dir(ansatz))

    estimator = Estimator()
    optimizer = COBYLA()  # read paper
    vqe = VQE(estimator, ansatz, optimizer)
    result = vqe.compute_minimum_eigenvalue(hamiltonian)

    optimal_state = ansatz.assign_parameters(result.optimal_parameters)
    optimal_state.measure_all()

    return optimal_state, result.optimal_value, n_entanglements


def evaluate_vqe_performance(optimal_value, target_value=None):
    """Evaluate and print the performance of VQE."""
    print(f"Optimal Energy: {optimal_value:.4f}")
    if target_value is not None:
        print(f"Target Energy: {target_value:.4f}")
        print(f"Energy Difference: {abs(optimal_value - target_value):.4f}")


def log_vqe_results(nodes, connectivity, reps, optimal_value, entangle_probability, entanglements,
                    file_path, target_value=None):
    """Log the VQE results to a CSV file."""
    fieldnames = ['nodes', 'connectivity', 'reps', 'optimal_value', 'target_value', 'energy_difference',
                  'entangle_prob', 'n_entanglements']
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
            'reps': reps,
            'optimal_value': optimal_value,
            'target_value': target_value if target_value is not None else 'N/A',
            'energy_difference': energy_difference,
            'entangle_prob': entangle_probability,
            'n_entanglements': entanglements,
        })


def max_cut_random_graph(nodes, connectivity, maximum_weight, reps, entangle_probability, filepath):
    """Solve the MaxCut problem on a randomly generated graph using VQE."""
    # Generate a random graph with specified number of nodes and connectivity
    num_nodes = nodes
    connectivity_prob = connectivity  # Probability of edge between nodes

    # Generate and convert to weighted edges
    random_graph = generate_random_graph(num_nodes, connectivity_prob, maximum_weight)
    weighted_edges = construct_weighted_edges_from_graph(random_graph)

    optimal_state, optimal_value, entanglements = solve_maxcut(weighted_edges, num_nodes, reps, entangle_probability)

    distribution, solution = get_solution(optimal_state)
    plot_solution(distribution, solution, "histogram_max_cut.png")
    evaluate_vqe_performance(optimal_value, target_value=-10)
    log_vqe_results(nodes, connectivity, reps, optimal_value, entangle_probability, entanglements, filepath,
                    target_value=None)
