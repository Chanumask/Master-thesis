import numpy as np

from max_cut import max_cut_random_graph
import pandas as pd


def run_multiple_iterations(nodes, connectivity, max_weight, ansatz_reps, entangle_probability, iterations, filepath):
    """Run multiple iterations of the MaxCut problem to collect statistical data."""
    for i in range(iterations):
        print(f"Running iteration {i + 1}/{iterations}")
        max_cut_random_graph(nodes, connectivity, max_weight, ansatz_reps, entangle_probability, filepath)


def evaluate_csv(filepath):
    data = pd.read_csv(filepath)

    average_optimal_values = data.groupby('n_entanglements')['optimal_value'].mean()
    sample_counts = data['n_entanglements'].value_counts().sort_index()

    print("Average Optimal Value per Number of Entanglements:")
    print(average_optimal_values)
    print("\nSample Count per Number of Entanglements:")
    print(sample_counts)


if __name__ == '__main__':
    np.random.seed(1)

    run_multiple_iterations(
        nodes=10,
        connectivity=0.8,
        max_weight=10,
        ansatz_reps=3,
        entangle_probability=1.0,
        iterations=1,
        filepath="vqe_results.csv")
    # evaluate_csv("vqe_results.csv")
