import os
from itertools import product

import numpy as np

from eval import entanglement_capability
from max_cut import generate_graph_from_config
from vqe import run_vqe_for_max_cut
from plotting import plot_metrics_across_run_from_csv, plot_energy_history_from_csv, read_all_runs, extract_metrics, \
    plot_average_iteration_time, read_and_average_experiments, plot_combined_expressibility_entanglement, \
    plot_combined_approximation_ratio, read_and_average_energy_histories, plot_averaged_energy_histories, check_outliers
from util import load_config, log_vqe_results, generate_deterministic_seeds, create_run_folder, \
    create_experiment_folder


def run_experiment(problem_name, config_file_name, logging, plotting):
    """
    Run experiments, handling both single runs and experiments with list parameters.
    If no parameters in the config are lists, the single run is treated as its own experiment.
    """
    # Load the configuration
    config = load_config(f'conf_files/{problem_name}/{config_file_name}.yaml')
    list_parameters = {
        key: value
        for section in ['vqe_parameters', 'max_cut']
        for key, value in config.get(section, {}).items()
        if isinstance(value, list)
    }
    # Create an experiment folder for this set of runs
    experiment_folder = create_experiment_folder()

    if list_parameters:
        # Prepare combinations of parameters if there are lists
        keys, values = zip(*list_parameters.items())
        combinations = [dict(zip(keys, combination)) for combination in product(*values)]

        for i, combination in enumerate(combinations):
            print(f"Running experiment ({i+1}/{len(combinations)})with parameters: {combination}")

            # Create a run folder inside the experiment folder
            run_folder = create_run_folder(experiment_folder)

            # Update the configuration with the current combination
            updated_config = config.copy()
            for key, value in combination.items():
                if key in config['vqe_parameters']:
                    updated_config['vqe_parameters'][key] = value
                elif key in config['max_cut']:
                    updated_config['max_cut'][key] = value

            # Pass the run folder to the single-run experiment
            run_single_experiment(updated_config, logging, plotting, run_folder)
    else:
        # Single run treated as its own experiment
        print("Running single-run experiment with the given configuration")

        # Create a single run folder inside the experiment folder
        run_folder = create_run_folder(experiment_folder)
        run_single_experiment(config, logging, plotting, run_folder)

    return experiment_folder


def run_single_experiment(config, logging, plotting, run_folder):
    """Run the VQE experiment based on the given configuration."""
    problem_class = config['problem_class']

    base_seed = config['base_seed']
    num_seeds = config['num_seeds']
    iterations = config['iterations']
    seeds = generate_deterministic_seeds(base_seed, num_seeds)

    result_list = []

    if problem_class == 'max_cut':
        for seed in seeds:
            np.random.seed(seed)  # Set the random seed for this iteration
            print(f"Running with seed: {seed}")

            for i in range(iterations):
                print(f"Running iteration {i + 1}/{iterations}")
                graph = generate_graph_from_config(config)
                result = run_vqe_for_max_cut(graph, config, seed, i)
                result_list.append(result)

                if logging:
                    # Save results to the appropriate run folder
                    log_vqe_results(result, config, os.path.join(run_folder, f"results_{config_name}.csv"))

            if plotting:
                # Generate plots for this run
                plot_metrics_across_run_from_csv(os.path.join(run_folder, f"results_{config_name}.csv"), run_folder)
                plot_energy_history_from_csv(os.path.join(run_folder, f"results_{config_name}.csv"), run_folder)
    else:
        raise ValueError("Unsupported problem class in configuration.")



if __name__ == '__main__':
    problem_type = "max_cut"
    config_name = "fully_connected"

    # experiment_path = run_experiment(problem_type, config_name, logging=True, plotting=True)
    experiment_path = "FINAL_RESULTS/parametrize/approx"


    # metrics_df = read_all_runs(config_name,experiment_path,[61])
    # plot_average_iteration_time(metrics_df, experiment_path)

    ### AVERAGED EXPERIMENTS

    # averaged_metrics_df = read_and_average_experiments(experiment_path, config_name, 'param_theta')   # param_theta , dropout_ratio

    # plot_combined_expressibility_entanglement(averaged_metrics_df, x_column='param_theta', experiment_folder=experiment_path)

    # plot_combined_approximation_ratio(
    #     metrics_df=averaged_metrics_df,
    #     x_column ='param_theta',
    #     experiment_folder=experiment_path)


    # CONVERGENCE

    df = read_and_average_energy_histories(experiment_path, "param_theta")
    # check_outliers(df)
    plot_averaged_energy_histories(df, "param_theta", experiment_path)
