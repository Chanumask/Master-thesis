import numpy as np

import networkx as nx
from qiskit import QuantumCircuit
from qiskit_optimization.applications import Maxcut
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.minimum_eigensolvers import VQE

from util import Result, log_vqe_results
from eval import evaluate_vqe_performance, get_solution, plot_solution
from vqe import add_cyclic_entanglement


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

    # Log results to CSV
    result = Result(
        nodes=g.number_of_nodes(),
        graph=g,  # g.edges() ?
        reps=reps,
        optimal_value=optimal_value,
        distribution=distribution,
        solution=solution,
        entangle_ratio=entangle_ratio,
        entanglements=entanglements,
        filepath=filepath
    )

    result.compute_energy_difference()

    return result


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


###########################

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
