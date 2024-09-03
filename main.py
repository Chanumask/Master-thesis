import numpy as np

from max_cut import generate_graph_from_config, max_cut_given_graph
from util import load_config, log_vqe_results


def run_experiment(config_file, logging, plotting):
    """Run the VQE experiment based on the configuration."""
    problem_class = config['problem_class']

    if problem_class == 'max_cut':
        graph = generate_graph_from_config(config_file)
        ansatz_layers = config['vqe_parameters']['ansatz_layers']
        entangle_ratio = config['vqe_parameters']['entangle_ratio']
        iterations = config['vqe_parameters']['iterations']
        filepath = config['vqe_parameters']['filepath']

        result_list = []
        for i in range(iterations):
            print(f"Running iteration {i + 1}/{iterations}")
            result = max_cut_given_graph(graph, ansatz_layers, entangle_ratio, filepath)
            result_list.append(result)

            if logging:
                log_vqe_results(result, filepath)
    else:
        raise ValueError("Unsupported problem class in configuration.")


if __name__ == '__main__':
    np.random.seed(1)

    config = load_config('conf_files/max_cut/fully_connected.yaml')

    save_results = True
    plot_results = False
    run_experiment(config, logging=True, plotting=False)

    # evaluate_csv("vqe_results.csv")
