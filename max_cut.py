import os
import subprocess

import networkx as nx
import numpy as np
import itertools

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from qiskit.primitives import Estimator
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA, SPSA, NELDER_MEAD
from qiskit_optimization.applications import Maxcut

from eval import get_solution, calculate_entanglement_metrics, calculate_cut_cost
from util import Result
from vqe import add_cyclic_entanglement, create_hea
from util import generate_qpic_circuit

global_energy_history = []


def brute_force_max_cut(graph):
    """Brute-force method to find the optimal Max-Cut value for a given graph."""
    num_nodes = graph.number_of_nodes()
    max_cut_value = float('-inf')

    # Iterate through all possible 2^n binary configurations of the nodes
    for cut in itertools.product([0, 1], repeat=num_nodes):
        cut_value = 0

        # Calculate the weight of the cut for the current configuration
        for u, v, weight in graph.edges(data='weight'):
            if cut[u] != cut[v]:  # u and v are in different partitions
                cut_value += weight

        max_cut_value = max(max_cut_value, cut_value)

    return max_cut_value


def run_vqe_for_max_cut(g, config, seed):
    """Solve the MaxCut problem on a given NetworkX graph using VQE."""
    global global_energy_history
    global_energy_history = []

    qubits = g.number_of_nodes()
    hamiltonian, offset = get_hamiltonian(g)

    ansatz, vqe = create_vqe(config, qubits)

    vqe.callback = vqe_callback

    result = vqe.compute_minimum_eigenvalue(hamiltonian)
    optimized_ansatz = ansatz.assign_parameters(result.optimal_parameters)

    entanglement_metrics = calculate_entanglement_metrics(optimized_ansatz)
    # print(entanglement_metrics)
    distribution, solution = get_solution(optimized_ansatz)
    bitstring_solution = solution[0]  # the most probable solution
    cut_cost = calculate_cut_cost(g, bitstring_solution)  # Calculate the cut cost based on the solution
    optimal_cut_value = brute_force_max_cut(g)

    # Log results to CSV
    result = Result(
        graph_edges=list(g.edges()),
        optimal_value=result.optimal_value,
        distribution=distribution,
        solution=solution,
        cut_cost=cut_cost,
        brute_forced_cost=optimal_cut_value,
        entanglement_metrics=entanglement_metrics,
        energy_history=global_energy_history,
        seed=seed,
    )
    result.compute_cut_cost_difference()
    result.compute_approximation_ratio()
    print(f"approx. ratio: {result.approximation_ratio}")
    return result


def get_hamiltonian(nxgraph):
    """Generate the Hamiltonian for the MaxCut problem."""
    maxcut = Maxcut(nxgraph)
    quadratic = maxcut.to_quadratic_program()
    return quadratic.to_ising()


def create_vqe(config, qubits):
    entangle_ratio = config['vqe_parameters']['entangle_ratio']
    ansatz_depth = config['vqe_parameters']['ansatz_layers']
    ansatz_type = config['vqe_parameters'].get('ansatz_type', 'RealAmplitudes')
    plot_ansatz = config['vqe_parameters']['plot_ansatz']
    optimizer_choice = config['vqe_parameters'].get('optimizer', 'COBYLA')

    # Ansatz selection
    if ansatz_type == 'RealAmplitudes':
        ansatz = RealAmplitudes(num_qubits=qubits, reps=ansatz_depth)
    elif ansatz_type == 'EfficientSU2':
        ansatz = EfficientSU2(num_qubits=qubits, reps=ansatz_depth)
    elif ansatz_type == 'HEA':
        ansatz = create_hea(qubits, ansatz_depth)
    else:
        raise ValueError(f"Unsupported ansatz type: {ansatz_type}")

    if plot_ansatz:
        qpic_str = generate_qpic_circuit(ansatz)
        print(qpic_str)
        with open('circuit.qpic', 'w', encoding='utf-8') as file:
            file.write(qpic_str)
            quit()
        # output_file = "circuit.png"
        # try:
        #     # Run qpic to generate a PNG
        #     subprocess.run(["qpic", "-f", "png", qpic_file], check=True)
        #
        #     from PIL import Image
        #     img = Image.open(output_file)
        #     img.show()
        #
        # except subprocess.CalledProcessError as e:
        #     print(f"Error generating the circuit image: {e}")
        # finally:
        #     os.remove(qpic_file)

    # Add Entanglement or introduce dropout
    full_circuit = QuantumCircuit(qubits)
    # full_circuit, n_entanglements = add_cyclic_entanglement(full_circuit, entangle_ratio)
    full_circuit.compose(ansatz, inplace=True)
    # print(full_circuit.decompose().draw())

    estimator = Estimator()

    # Select the optimizer
    if optimizer_choice == 'COBYLA':
        optimizer = COBYLA()
    elif optimizer_choice == 'SPSA':
        optimizer = SPSA()
    elif optimizer_choice == 'NELDER_MEAD':
        optimizer = NELDER_MEAD()
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_choice}")

    vqe = VQE(estimator, full_circuit, optimizer)

    return ansatz, vqe


def vqe_callback(eval_count, params, energy, metadata):
    """Callback function to store the energy at each iteration.""" 
    global global_energy_history
    if eval_count % 10 == 0:    # only save every 10th value
        global_energy_history.append(float(round(energy,4)))

##################


def generate_graph_from_config(config):
    """Generate the graph based on the configuration."""
    problem_class = config['problem_class']

    if problem_class == 'max_cut':
        graph_type = config['max_cut']['graph_type']
        nodes = config['max_cut']['nodes']

        if graph_type == 'fully_connected':
            max_weight = config['max_cut']['max_weight']
            graph = generate_fully_connected_graph(nodes, max_weight)

        elif graph_type == 'random':
            connectivity = config['max_cut']['connectivity']
            max_weight = config['max_cut']['max_weight']
            graph = generate_random_graph(nodes, connectivity, max_weight)

        elif graph_type == '3-regular':
            degree = config['max_cut']['degree']
            graph = generate_regular_graph(degree, nodes)

        else:
            raise ValueError("Unknown graph type in conf file.")

    else:
        raise ValueError("Unknown problem class in conf file.")

    return graph


def generate_fully_connected_graph(num_nodes, max_weight):
    # Create a fully connected graph (complete graph) with 8 nodes
    g = nx.Graph()

    # Add all possible edges between the 8 nodes with random weights
    nodes = range(num_nodes)
    for i in nodes:
        for j in nodes:
            if i < j:  # To ensure each edge is added only once
                weight = np.random.randint(1, max_weight + 1)
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


def generate_regular_graph(degree, nodes):
    g = nx.random_regular_graph(degree, nodes)  # n-regular graph, unit edge weights

    nx.random_regular_graph(degree, nodes)
    for (u, v) in g.edges():
        g.edges[u, v]['weight'] = np.random.randint(1, 11)
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
