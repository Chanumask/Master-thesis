import networkx as nx
import numpy as np
import itertools

from qiskit_optimization.applications import Maxcut

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
