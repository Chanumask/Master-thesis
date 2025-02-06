import os
from itertools import product

import numpy as np

from max_cut import generate_graph_from_config
from vqe import run_vqe_for_max_cut
from plotting import plot_metrics_across_run_from_csv, plot_energy_history_from_csv, read_all_runs, extract_metrics, \
    plot_metrics_with_grouping, plot_expressibility_entanglement, plot_average_iteration_time, \
    read_and_average_experiments
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
    experiment_path = run_experiment(problem_type, config_name, logging=True, plotting=True)
    # experiment_path = "FINAL_RESULTS/DROPOUT_STUDY/10_seeds/"

    # parameter_column = ['ansatz_layers','dropout_ratio', 'param_theta']  # ansatz_layers, dropout_ratio, param_theta

    # combined_df = read_all_runs(config_name, experiment_ids = [1])
    # metrics_df = extract_metrics(combined_df)
    # plot_average_metrics(metrics_df, parameter_column, experiment_path)
    # plot_approximation_vs_entanglement(metrics_df, experiment_path)
    # plot_metrics_with_dropout(metrics_df, parameter_column='ansatz_layers', group_column='dropout_ratio', experiment_folder=experiment_path)

    # plot_metrics_with_grouping(
    #     metrics_df=metrics_df.drop(columns=['entangling_capability', 'expressibility']),
    #     parameter_column='dropout_ratio',      # x axis
    #     group_column='ansatz_type',            # Grouping by ansatz type
    #     metrics_to_plot=['approximation_ratio'],
    #     experiment_folder=experiment_path
    # )

    # plot_expressibility_entanglement(
    #     metrics_df=metrics_df,
    #     x_column='ansatz_layers',      # Parameter to vary on the x-axis: ansatz_layers, dropout_ratio, param_theta
    #     group_column='ansatz_type',   # Grouping by ansatz type or other variables
    #     experiment_folder=experiment_path  # Path to save the plots
    # )

    # plot_average_iteration_time(metrics_df, experiment_path)

    # averaged_metrics_df = read_and_average_experiments(experiment_path, config_name)

    # Plot averaged results
    # plot_metrics_with_grouping(
    # metrics_df=averaged_metrics_df.drop(columns=['entangling_capability', 'expressibility']),
    # parameter_column='dropout_ratio',
    # group_column='ansatz_type',
    # metrics_to_plot=['approximation_ratio'],
    # experiment_folder=experiment_path
    # )
