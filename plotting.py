import csv
import os

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')
import ast


def read_all_runs(config_name, experiment_ids):
    """
    Read CSV files from runs within specified experiments and return combined dataframe.
    Args:
        config_name (str): Configuration name for identifying CSV files.
        experiment_ids (list): List of experiment indices to process.

    Returns:
        pd.DataFrame: Combined dataframe of all runs within the experiments.
    """
    all_data_frames = []

    for experiment_id in experiment_ids:
        experiment_folder = f"results/Experiment_{experiment_id}"

        if os.path.exists(experiment_folder):
            # Iterate over all runs within the experiment folder
            for run_folder_name in os.listdir(experiment_folder):
                run_folder = os.path.join(experiment_folder, run_folder_name)

                if os.path.isdir(run_folder) and run_folder_name.startswith("Run_"):
                    csv_file = os.path.join(run_folder, f"results_{config_name}.csv")

                    if os.path.exists(csv_file):
                        df = pd.read_csv(csv_file)
                        df['experiment_id'] = experiment_id
                        df['run_id'] = int(run_folder_name.replace("Run_", ""))
                        all_data_frames.append(df)
                    else:
                        print(f"CSV file not found: {csv_file}")
        else:
            print(f"Experiment folder not found: {experiment_folder}")

    if all_data_frames:
        combined_data = pd.concat(all_data_frames, ignore_index=True)
        return combined_data
    else:
        raise ValueError("No valid CSV files found for the given experiment IDs.")


def extract_metrics(data_frame):
    """Extract entanglement metrics and ansatz_layers from the DataFrame."""
    metrics_list = []
    for _, row in data_frame.iterrows():
        metrics_str = row['entanglement_metrics']
        ansatz_type = row.get('ansatz_type', None)
        dropout_ratio = row.get('dropout_ratio', None)
        ansatz_layers = row.get('ansatz_layers', None)
        ent_gate = row.get('ent_gate', None)
        param_theta = row.get('param_theta', None)
        approximation_ratio = row.get('approximation_ratio', None)

        # Safely evaluate param_theta if it's a string expression
        if isinstance(param_theta, str):
            try:
                param_theta = eval(param_theta, {"np": np})
            except Exception as e:
                print(f"Error evaluating param_theta: {e}")
                param_theta = None

        safe_metrics_str = metrics_str.replace('np.float64', '').replace('(', '').replace(')', '')
        safe_metrics_str = safe_metrics_str.replace('{\'', '{"').replace('\':', '":').replace(', \'', ', "')
        try:
            # Convert string to dictionary
            metrics_data = ast.literal_eval(safe_metrics_str)
            e_per_qubit_value = metrics_data[1]['e_per_qubit']
            meyer_wallach_value = metrics_data[1]['meyer_wallach']
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing metrics: {e}")
            e_per_qubit_value = None
            meyer_wallach_value = None

        metrics_list.append({
            'ansatz_type': ansatz_type,
            'ansatz_layers': ansatz_layers,
            'ent_gate': ent_gate,
            'dropout_ratio': dropout_ratio,
            'param_theta': param_theta,  # Now numeric if evaluable
            'e_per_qubit': e_per_qubit_value,
            'meyer_wallach': meyer_wallach_value,
            'approximation_ratio': approximation_ratio
        })
    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df



def plot_average_metrics(metrics_df, parameter_column, experiment_folder):
    """Plot average entanglement metrics vs. a varying parameter and save to the experiment folder."""
    average_metrics = metrics_df.groupby(parameter_column).mean().reset_index()
    print(average_metrics)
    metrics_to_plot = ['e_per_qubit', 'meyer_wallach', 'approximation_ratio']

    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(average_metrics[parameter_column], average_metrics[metric], marker='o', label=f'Average {metric}')
        plt.title(f"Average {metric} vs. {parameter_column}")
        plt.xlabel(f"{parameter_column.replace('_', ' ').title()}")
        plt.ylabel(f"Average {metric}")
        plt.legend()
        plt.grid(True)

        # Save the plot to the experiment folder
        save_path = os.path.join(experiment_folder, f"{metric}_vs_{parameter_column}.png")
        plt.savefig(save_path)
        plt.close()


def plot_approximation_vs_entanglement(metrics_df, experiment_folder):
    """Plot approximation ratio vs. entanglement metrics and save to the experiment folder."""
    metrics_to_plot = ['e_per_qubit', 'meyer_wallach']

    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(metrics_df[metric], metrics_df['approximation_ratio'], alpha=0.7,
                    label=f'Approximation Ratio vs. {metric}')
        plt.title(f"Approximation Ratio vs. {metric}")
        plt.xlabel(f"{metric.replace('_', ' ').title()}")
        plt.ylabel("Approximation Ratio")
        plt.legend()
        plt.grid(True)

        # Save the plot to the experiment folder
        save_path = os.path.join(experiment_folder, f"approximation_vs_{metric}.png")
        plt.savefig(save_path)
        plt.close()


def plot_energy_history_from_csv(filepath, out_path):
    """Plot the energy history from the result CSV."""
    with open(filepath, mode='r') as file:
        reader = csv.DictReader(file)

        energy_histories = []
        run_labels = []

        for row in reader:
            eh_string = row.get('energy_history', '')
            if eh_string:
                eh = ast.literal_eval(eh_string)
                energy_histories.append(eh)
                run_labels.append(f"Run {len(run_labels) + 1}")

        plt.figure(figsize=(10, 6))

        for i, energy_history in enumerate(energy_histories):
            steps = [x * 10 for x in range(1, len(energy_history) + 1)]
            plt.plot(steps, energy_history, marker='o', linestyle='-', label=run_labels[i])

        plt.title('Energy History for Each Run')
        plt.xlabel('Step')
        plt.ylabel('Energy')
        plt.legend(title='Runs')
        plt.grid(True)
        plt.savefig(f"{out_path}/energy_history_plot.png", format='png')
        plt.close()


def save_plots(all_data, metrics_to_plot, resultpath):
    base_dir = resultpath

    # Plot and save each metric
    for metric in metrics_to_plot:
        metric_data = all_data[metric]
        if any(d is not None for d in metric_data):
            plt.figure(figsize=(10, 6))
            plt.plot(metric_data, label=metric)
            plt.title(f"{metric} across experiments")
            plt.xlabel("Experiment index")
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)

            # Save plot as PNG file in current run folder
            save_path = os.path.join(base_dir, f"{metric}.png")
            plt.savefig(save_path)
            plt.close()


def plot_metrics_across_run_from_csv(csvpath, results_folder):
    """Read CSV files from the results folder and plot relevant metrics."""
    metrics_to_plot = ['meyer_wallach']
    all_data = {metric: [] for metric in metrics_to_plot}
    filenames = []

    data_df = pd.read_csv(csvpath)
    # Extract entanglement metrics from each row
    for _, row in data_df.iterrows():
        metrics_str = row['entanglement_metrics']
        metrics_data = eval(metrics_str)

        for metric in metrics_to_plot:
            if metric in metrics_data:
                all_data[metric].append(metrics_data[metric])
            else:
                all_data[metric].append(None)  # handle missing values

    save_plots(all_data, metrics_to_plot, results_folder)


def plot_metrics_with_grouping(metrics_df, parameter_column, group_column, metrics_to_plot, experiment_folder):
    """
    Plot average entanglement metrics vs. ansatz_layers with different colors for each dropout value and save the plot.
    Args:
        metrics_df (pd.DataFrame): DataFrame with metrics data.
        parameter_column (str): The column name for the x-axis (e.g., 'ansatz_layers').
        group_column (str): The column to group by for color coding (e.g., 'dropout_ratio').
        experiment_folder (str): Path to the experiment folder where the plot will be saved.
        metrics_to_plot (list): list of metrics to generate plots for on the vertical axis
    """
    # Group the data by the parameter column and group column
    print(metrics_df.columns)
    grouped_metrics = metrics_df.groupby([parameter_column, group_column]).mean().reset_index()

    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        # Use seaborn for easy color handling
        sns.lineplot(
            data=grouped_metrics,
            x=parameter_column,
            y=metric,
            hue=group_column,
            marker="o",
            palette="tab10"
        )
        plt.title(f"Average {metric} vs. {parameter_column} (Grouped by {group_column})")
        plt.xlabel(f"{parameter_column.replace('_', ' ').title()}")
        plt.ylabel(f"Average {metric}")
        plt.legend(title=group_column.replace('_', ' ').title())
        plt.grid(True)

        # Save the plot to the experiment folder
        save_path = os.path.join(experiment_folder, f"{metric}_vs_{parameter_column}_by_{group_column}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")
