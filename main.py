import networkx as nx
import numpy as np

from eval import evaluate_csv
from max_cut import max_cut_random_graph, max_cut_given_graph, generate_fully_connected_graph


def run_multiple_iterations_random_graph(nodes, connectivity, max_weight, ansatz_layers, entangle_ratio, iterations,
                                         filepath):
    """Run multiple iterations of the MaxCut problem to collect statistical data."""
    for i in range(iterations):
        print(f"Running iteration {i + 1}/{iterations}")
        max_cut_random_graph(nodes, connectivity, max_weight, ansatz_layers, entangle_ratio, filepath)


def run_multiple_iterations_given_graph(graph, ansatz_layers, entangle_ratio, iterations, filepath):
    """Run multiple iterations of the MaxCut problem on a given graph to collect statistical data."""
    for i in range(iterations):
        print(f"Running iteration {i + 1}/{iterations}")
        max_cut_given_graph(graph, ansatz_layers, entangle_ratio, filepath)


if __name__ == '__main__':
    np.random.seed(1)

    # run_multiple_iterations_random_graph(
    #     nodes=10,
    #     connectivity=0.8,
    #     max_weight=10,
    #     ansatz_layers=1,
    #     entangle_ratio=1.0,
    #     iterations=1,
    #     filepath="results_random_graph.csv")

    g = generate_fully_connected_graph(8, 10)

    # Run multiple iterations on the given graph
    run_multiple_iterations_given_graph(
        graph=g,
        ansatz_layers=1,
        entangle_ratio=1.0,
        iterations=1,
        filepath="results_given_graph.csv")

    # evaluate_csv("vqe_results.csv")
