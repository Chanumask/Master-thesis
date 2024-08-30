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

from util import get_solution, plot_solution, create_hea, add_cyclic_entanglement


def generate_fully_connected_graph(num_nodes, max_weight):
    # Create a fully connected graph (complete graph) with 8 nodes
    g = nx.Graph()

    # Add all possible edges between the 8 nodes with random weights
    nodes = range(num_nodes)
    for i in nodes:
        for j in nodes:
            if i < j:  # To ensure each edge is added only once
                weight = np.random.randint(1, max_weight+1)
                g.add_edge(i, j, weight=weight)

    return g


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


def solve_maxcut(graph, qubits, ansatz_depth, entangle_ratio):
    """Solve the MaxCut problem for a provided graph using VQE and return the solution."""

    hamiltonian, offset = get_hamiltonian(graph)

    ansatz = RealAmplitudes(num_qubits=qubits, reps=ansatz_depth)
    # ansatz = create_hea(qubits, ansatz_depth)

    full_circuit = QuantumCircuit(qubits)
    full_circuit, n_entanglements = add_cyclic_entanglement(full_circuit, entangle_ratio)
    full_circuit.compose(ansatz, inplace=True)

    # print(full_circuit.decompose().draw())
    # print(dir(ansatz))

    estimator = Estimator()
    optimizer = COBYLA()  # read paper
    vqe = VQE(estimator, full_circuit, optimizer)
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


def max_cut_random_graph(nodes, connectivity, maximum_weight, reps, entangle_ratio, filepath):
    """Solve the MaxCut problem on a randomly generated graph using VQE."""

    # Generate a random graph with specified number of nodes and connectivity
    random_graph = generate_random_graph(nodes, connectivity, maximum_weight)
    # weighted_edges = construct_weighted_edges_from_graph(random_graph)

    optimal_state, optimal_value, entanglements = solve_maxcut(random_graph, nodes, reps, entangle_ratio)

    distribution, solution = get_solution(optimal_state)
    plot_solution(distribution, solution, "histogram_max_cut.png")
    evaluate_vqe_performance(optimal_value, target_value=-10)  # how to find out target value?
    log_vqe_results(nodes, connectivity, reps, optimal_value, entangle_ratio, entanglements, filepath,
                    target_value=None)


def max_cut_given_graph(g, reps, entangle_ratio, filepath):
    """Solve the MaxCut problem on a given NetworkX graph using VQE."""

    optimal_state, optimal_value, entanglements = solve_maxcut(g, g.number_of_nodes(), reps, entangle_ratio)

    distribution, solution = get_solution(optimal_state)
    plot_solution(distribution, solution, "histogram_max_cut.png")

    # Evaluate performance (set target_value to None if unknown)
    target_value = None  # Replace with actual target value if known
    evaluate_vqe_performance(optimal_value, target_value)

    # Log results to CSV
    log_vqe_results(
        g.number_of_nodes(),
        None,  # No connectivity value available since the graph is given directly
        reps,
        optimal_value,
        entangle_ratio,
        entanglements,
        filepath,
        target_value
    )
