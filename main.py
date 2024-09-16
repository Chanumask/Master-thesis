import numpy as np

from max_cut import generate_graph_from_config, run_vqe_for_max_cut
from util import load_config, log_vqe_results


def run_experiment(config_file, logging, plotting):
    """Run the VQE experiment based on the configuration."""
    problem_class = config['problem_class']

    if problem_class == 'max_cut':
        graph = generate_graph_from_config(config_file)
        iterations = config['iterations']
        filepath = config['filepath']
        result_list = []
        for i in range(iterations):
            print(f"Running iteration {i + 1}/{iterations}")
            result = run_vqe_for_max_cut(graph, config_file)
            result_list.append(result)

            if logging:
                log_vqe_results(result, filepath)
    else:
        raise ValueError("Unsupported problem class in configuration.")


if __name__ == '__main__':
    np.random.seed(1)

    config = load_config('conf_files/max_cut/random.yaml')

    save_results = True
    plot_results = False
    run_experiment(config, logging=True, plotting=False)
