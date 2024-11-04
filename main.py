import numpy as np

from max_cut import generate_graph_from_config, run_vqe_for_max_cut
from plotting import plot_metrics_from_csv, plot_energy_history_from_csv
from util import load_config, log_vqe_results, generate_deterministic_seeds


def run_experiment(config_name, logging, plotting):
    """Run the VQE experiment based on the configuration."""
    config = load_config(f'conf_files/max_cut/{config_name}.yaml')
    problem_class = config['problem_class']

    filepath = config['filepath']

    base_seed = config['base_seed']
    num_seeds = config['num_seeds']
    iterations = config['iterations']
    seeds = generate_deterministic_seeds(base_seed, num_seeds)

    if problem_class == 'max_cut':
        result_list = []

        for seed in seeds:
            np.random.seed(seed)  # Set the random seed for this iteration
            print(f"Running with seed: {seed}")

            for i in range(iterations):
                print(f"Running iteration {i + 1}/{iterations}")
                graph = generate_graph_from_config(config)
                result = run_vqe_for_max_cut(graph, config, seed)
                result_list.append(result)

                if logging:
                    log_vqe_results(result, config, filepath)

                if plotting:
                    plot_metrics_from_csv('all', 'results')
                    plot_energy_history_from_csv(f"results/{config_name}.csv")

    else:
        raise ValueError("Unsupported problem class in configuration.")


if __name__ == '__main__':
    run_experiment(config_name= "fully_connected", logging=False, plotting=False)
