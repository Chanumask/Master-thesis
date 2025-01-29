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
        nodes = row.get('nodes', None)
        ansatz_type = row.get('ansatz_type', None)
        dropout_ratio = row.get('dropout_ratio', None)
        ansatz_layers = row.get('ansatz_layers', None)
        ent_gate = row.get('ent_gate', None)
        param_theta = row.get('param_theta', None)
        approximation_ratio = row.get('approximation_ratio', None)
        iteration_time = row.get('iter_time', None)


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
            e_per_qubit_value = metrics_data['e_per_qubit']
            meyer_wallach_value = metrics_data['meyer_wallach']
            ent_cap = metrics_data['entangling_capability']
            expressibility = metrics_data['expressibility']
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing metrics: {e}")
            e_per_qubit_value = None
            meyer_wallach_value = None
            ent_cap = None
            expressibility = None

        metrics_list.append({
            'nodes': nodes,
            'ansatz_type': ansatz_type,
            'ansatz_layers': ansatz_layers,
            'ent_gate': ent_gate,
            'dropout_ratio': dropout_ratio,
            'param_theta': param_theta,  # Now numeric if evaluable
            'e_per_qubit': e_per_qubit_value,
            'meyer_wallach': meyer_wallach_value,
            'approximation_ratio': approximation_ratio,
            'entangling_capability': ent_cap,
            'expressibility': expressibility,
            'iteration_time': iteration_time
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
    Plot average metrics vs. a parameter, grouped by another column.

    Args:
        metrics_df (pd.DataFrame): DataFrame with metrics data.
        parameter_column (str): Column name for the x-axis (e.g., 'ansatz_layers').
        group_column (str): Column to group by (e.g., 'dropout_ratio').
        metrics_to_plot (list): List of metrics to plot (y-axis).
        experiment_folder (str): Directory where the plots will be saved.
    """
    print(metrics_df.columns)

    # Filter only numeric columns for grouping and aggregation
    numeric_columns = metrics_df.select_dtypes(include=[np.number]).columns

    # Avoid duplicate column insertion after resetting the index
    grouped_metrics = metrics_df.groupby([parameter_column, group_column])[numeric_columns].mean()

    for metric in metrics_to_plot:
        if metric not in grouped_metrics.columns:
            print(f"Metric '{metric}' is not in the grouped data. Skipping...")
            continue

        plt.figure(figsize=(10, 6))
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


def plot_expressibility_entanglement(metrics_df, x_column, group_column, experiment_folder):
    """
    Plot expressibility and entangling capability metrics for varying parameters.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing the metrics data.
        x_column (str): The column name for the x-axis (e.g., 'ansatz_layers', 'dropout_ratio', 'param_theta').
        group_column (str): The column to group by for color coding (e.g., 'ansatz_type', 'dropout_ratio').
        experiment_folder (str): Path to the experiment folder where the plots will be saved.
    """
    print(metrics_df.columns)

    # Metrics to plot
    metrics_to_plot = ['expressibility', 'entangling_capability']

    # Filter numeric columns
    numeric_columns = metrics_df.select_dtypes(include=[np.number]).columns.tolist()

    # Ensure that `x_column` and `group_column` are included
    numeric_columns = list(set(numeric_columns + [x_column, group_column]))

    # Group the data using only numeric columns
    grouped_metrics = metrics_df[numeric_columns].groupby([x_column, group_column]).mean().reset_index()

    for metric in metrics_to_plot:
        if metric not in grouped_metrics.columns:
            print(f"Metric '{metric}' not found in grouped data. Skipping...")
            continue

        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=grouped_metrics,
            x=x_column,
            y=metric,
            hue=group_column,
            marker="o",
            palette="tab10"
        )
        plt.title(f"{metric.replace('_', ' ').title()} vs. {x_column.replace('_', ' ').title()} (Grouped by {group_column.replace('_', ' ').title()})")
        plt.xlabel(x_column.replace('_', ' ').title())
        plt.ylabel(metric.replace('_', ' ').title())
        plt.legend(title=group_column.replace('_', ' ').title(), loc="best")
        plt.grid(True)

        # Save the plot
        save_path = os.path.join(experiment_folder, f"{metric}_vs_{x_column}_by_{group_column}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")

def plot_average_iteration_time(metrics_df, experiment_folder):
    """
    Plot average iteration time for each run against the number of qubits.
    Group data by ansatz_type and ansatz_layers.
    Display the times in MM:SS format on the primary axis.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing metrics data.
        experiment_folder (str): Path to the folder where the plots will be saved.
    """
    # Ensure numeric columns
    metrics_df['nodes'] = pd.to_numeric(metrics_df['nodes'], errors='coerce')
    metrics_df['iteration_time'] = pd.to_numeric(metrics_df['iteration_time'], errors='coerce')

    # Take the absolute value of iteration_time
    metrics_df['iteration_time'] = metrics_df['iteration_time'].abs()

    # Drop rows with missing data
    metrics_df = metrics_df.dropna(subset=['nodes', 'iteration_time', 'ansatz_type', 'ansatz_layers'])

    # Calculate average times
    average_including_first = metrics_df.groupby(['nodes', 'ansatz_type', 'ansatz_layers'])['iteration_time'].mean()
    metrics_df_excluding_first = metrics_df.loc[
        metrics_df.groupby(['nodes', 'ansatz_type', 'ansatz_layers']).cumcount() > 0]
    average_excluding_first = metrics_df_excluding_first.groupby(['nodes', 'ansatz_type', 'ansatz_layers'])[
        'iteration_time'].mean()

    # Convert seconds to MM:SS format
    def format_seconds(seconds, add_strings=False):
        minutes = int(seconds // 60)
        sec = int(seconds % 60)
        if add_strings:
            return f"{minutes} min {sec} sec"
        return f"{minutes}:{sec:02d}"

    # Print average times in MM:SS format
    print("Average Iteration Time Including First Iteration:")
    print(average_including_first.apply(lambda x: format_seconds(x, add_strings=False)))

    print("\nAverage Iteration Time Excluding First Iteration:")
    print(average_excluding_first.apply(lambda x: format_seconds(x, add_strings=False)))

    def plot_average(df, title, filename):
        # Reset index for plotting
        df = df.reset_index()
        # Create a combined label for ansatz_type and ansatz_layers
        df['group'] = df['ansatz_type'] + ", Layers: " + df['ansatz_layers'].astype(str)

        # Define a formatter for the y-axis
        def seconds_to_mmss(x, _):
            return format_seconds(x, add_strings=False)

        # Plot the data
        fig, ax1 = plt.subplots(figsize=(10, 6))

        sns.lineplot(
            data=df,
            x='nodes',
            y='iteration_time',
            hue='group',
            marker='o',
            ax=ax1
        )
        ax1.set_ylabel("Average Iteration Time (MM:SS)")
        ax1.set_xlabel("Number of Qubits (nodes)")
        ax1.legend(title="Ansatz Type and Layers")
        ax1.grid(True)

        # Apply the custom formatter to the y-axis
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(seconds_to_mmss))

        # Adjust title and save the plot
        plt.title(title)
        save_path = os.path.join(experiment_folder, filename)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {save_path}")

    # Plot for averages including the first iteration
    plot_average(
        average_including_first,
        "Average Iteration Time (Including First Iteration) vs Number of Qubits",
        "avg_iter_time_including_first.png"
    )

    # Plot for averages excluding the first iteration
    plot_average(
        average_excluding_first,
        "Average Iteration Time (Excluding First Iteration) vs Number of Qubits",
        "avg_iter_time_excluding_first.png"
    )
