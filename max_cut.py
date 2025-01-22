import networkx as nx
import os
import numpy as np
import itertools

from qiskit_optimization.applications import Maxcut

from eval import calculate_cut_cost, get_probabilities, calculate_entanglement_metrics, compute_expressibility, \
    entanglement_capability
from util import Result, generate_qpic_circuit
from vqe import create_vqe


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


def run_vqe_for_max_cut(g, config, seed, num_iteration):
    """Solve the MaxCut problem on a given NetworkX graph using VQE."""
    # print("Starting VQE optimization for MAX-CUT problem")

    qubits = g.number_of_nodes()
    hamiltonian, offset = get_hamiltonian(g)

    global_energy_history = []
    ansatz, assigned_parameters, vqe = create_vqe(config, qubits, global_energy_history, num_iteration)

    # calculate pre-training metrics and plot ansatz
    expressibility_score = None
    entangling_capability = None

    if num_iteration == 0:
        ansatz_type = config['vqe_parameters']['ansatz_type']
        ansatz_layers = config['vqe_parameters']['ansatz_layers']
        entangle_method = config['vqe_parameters']['entangle_method']
        if config['eval']['ent_cap']:
            entangling_capability = entanglement_capability(ansatz)

        if config['eval']['qpic']:

            directory = "ansätze"
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Generate QPIC filename
            qpic_filename = f"{directory}/{ansatz_type}.qpic"

            # Generate QPIC file
            if not os.path.exists(qpic_filename):
                qpic_str = generate_qpic_circuit(ansatz)
                with open(qpic_filename, 'w', encoding='utf-8') as file:
                    file.write(qpic_str)
                print(f"Generated QPIC file for ansatz: {qpic_filename}")
            else:
                print(f"QPIC file already exists: {qpic_filename}")

        if config['eval']['expressibility']:
            # Determine the suffix based on the entangle method
            if entangle_method == 'dropout':
                suffix = f"d{config['vqe_parameters']['dropout_ratio']}"
            elif entangle_method == 'parametrize':
                param_theta = config['vqe_parameters']['param_theta']

                if 'pi' in param_theta:
                    # Extract everything after 'pi' (e.g., "/2" or "/4") and replace "/" with "_"
                    pi_part = param_theta.split('pi')[1].strip()  # Gets the part after 'pi'
                    sanitized_theta = f"pi{pi_part.replace('/', '_')}" if pi_part else "pi"
                else:
                    # If 'pi' is not in the string, keep the original value as the suffix
                    sanitized_theta = param_theta.replace("/", "_")
                suffix = f"p{sanitized_theta}"

            else:
                suffix = ""

            # Construct the expressibility filename
            expressibility_filename = f"{directory}/{ansatz_type}_{ansatz_layers}_{suffix}".strip("_")

            # Check if the file already exists
            if not os.path.exists(os.path.join(expressibility_filename, "_expressibility.png")):
                expressibility_score = compute_expressibility(ansatz, expressibility_filename)
                print(f"Generated expressibility graph: {expressibility_filename}")
            else:
                print(f"Expressibility graph already exists: {expressibility_filename}")

    # Run optimization
    vqe_result = vqe.compute_minimum_eigenvalue(hamiltonian)
    optimized_ansatz = ansatz.assign_parameters(vqe_result.optimal_parameters)

    probabilities = get_probabilities(optimized_ansatz)

    entanglement_metrics = calculate_entanglement_metrics(optimized_ansatz, probabilities)
    entanglement_metrics['expressibility'] = expressibility_score
    entanglement_metrics['entangling_capability'] = entangling_capability

    print(entanglement_metrics)

    solution = max(probabilities.items(), key=lambda x: x[1])
    bitstring_solution = solution[0]
    cut_cost = calculate_cut_cost(g, bitstring_solution)
    optimal_cut_value = brute_force_max_cut(g)

    # Log results to CSV
    result = Result(
        graph_edges=list(g.edges()),
        optimal_value=vqe_result.optimal_value,
        distribution=probabilities,
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
