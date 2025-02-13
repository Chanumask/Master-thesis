import csv
import os

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import label

matplotlib.use('Agg')
import ast


def read_all_runs(config_name, experiment_path, experiment_ids):
    """
    Read CSV files from runs within specified experiments and return combined dataframe.
    Args:
        config_name (str): Configuration name for identifying CSV files.
        experiment_path: base path of experiment runs
        experiment_ids (list): List of experiment indices to process.

    Returns:
        pd.DataFrame: Combined dataframe of all runs within the experiments.
    """
    all_data_frames = []

    for experiment_id in experiment_ids:
        experiment_folder = os.path.join(experiment_path, f"Experiment_{experiment_id}")
        print(experiment_folder)

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
            ent_cap = metrics_data.get('entangling_capability', None)
            expressibility = metrics_data.get('expressibility', None)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing metrics: {e}")
            ent_cap = None
            expressibility = None

        metrics_list.append({
            'nodes': nodes,
            'ansatz_type': ansatz_type,
            'ansatz_layers': ansatz_layers,
            'dropout_ratio': dropout_ratio,
            'param_theta': param_theta,  # Now numeric if evaluable
            'approximation_ratio': approximation_ratio,
            'entangling_capability': ent_cap,
            'expressibility': expressibility,
            'iteration_time': iteration_time
        })
    metrics_df = pd.DataFrame(metrics_list)
    print(metrics_df.columns)
    return metrics_df


def read_and_average_experiments(experiment_path, config_name, x_column):
    """
    Reads all experiments within a directory and aggregates their metrics,
    computing mean, standard deviation, confidence intervals, and quantiles.

    Args:
        experiment_path (str): Path containing all experiment subdirectories.
        config_name (str): Configuration name for identifying CSV files.
        x_column: 'dropout' or 'param_theta'

    Returns:
        pd.DataFrame: DataFrame containing aggregated metrics with variability estimates.
    """
    all_experiment_ids = []

    # Identify available experiments
    for folder_name in os.listdir(experiment_path):
        if folder_name.startswith("Experiment_"):
            try:
                experiment_id = int(folder_name.replace("Experiment_", ""))
                all_experiment_ids.append(experiment_id)
            except ValueError:
                continue

    if not all_experiment_ids:
        raise ValueError(f"No valid experiments found in {experiment_path}")

    print("Processing experiments:", all_experiment_ids)

    # Read data from all experiments
    combined_df = read_all_runs(config_name, experiment_path=experiment_path, experiment_ids=all_experiment_ids)
    metrics_df = extract_metrics(combined_df)

    # Define aggregation functions with safe handling for missing values
    def safe_agg(column):
        if column.dropna().empty:  # Check if all values are NaN
            return np.nan
        return column.mean()  # Default: return the mean

    def safe_std(column):
        if column.dropna().empty:
            return np.nan
        return column.std()

    def safe_ci(column):
        if column.dropna().empty or len(column.dropna()) < 2:  # Need at least 2 samples for CI
            return np.nan
        return 1.645 * sum(column.dropna())

    def safe_quantile(column, q):
        if column.dropna().empty:
            return np.nan
        return column.quantile(q)

    # Compute mean, standard deviation, CI, and quantiles for each metric independently
    aggregated_metrics = metrics_df.groupby([x_column, 'ansatz_type', 'ansatz_layers']).agg(
        mean_expressibility=('expressibility', safe_agg),
        std_expressibility=('expressibility', safe_std),
        ci_expressibility=('expressibility', safe_ci),
        q10_expressibility=('expressibility', lambda x: safe_quantile(x, 0.1)),
        q90_expressibility=('expressibility', lambda x: safe_quantile(x, 0.9)),

        mean_entangling=('entangling_capability', safe_agg),
        std_entangling=('entangling_capability', safe_std),
        ci_entangling=('entangling_capability', safe_ci),
        q10_entangling=('entangling_capability', lambda x: safe_quantile(x, 0.1)),
        q90_entangling=('entangling_capability', lambda x: safe_quantile(x, 0.9)),

        mean_approximation=('approximation_ratio', safe_agg),
        std_approximation=('approximation_ratio', safe_std),
        ci_approximation=('approximation_ratio', safe_ci),
        q10_approximation=('approximation_ratio', lambda x: safe_quantile(x, 0.1)),
        q90_approximation=('approximation_ratio', lambda x: safe_quantile(x, 0.9))
    ).reset_index()

    return aggregated_metrics


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
        plt.title(
            f"{metric.replace('_', ' ').title()} vs. {x_column.replace('_', ' ').title()} (Grouped by {group_column.replace('_', ' ').title()})")
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


def plot_combined_expressibility_entanglement(metrics_df, x_column, experiment_folder):
    """
    Plot expressibility and entangling capability with error bands, grouped by ansatz type and layers.
    It generates both a single grouped plot and separate subplots per ansatz type.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing the metrics with variability measures.
        x_column (str): The column name for the x-axis (e.g., 'dropout_ratio', 'param_theta').
        experiment_folder (str): Path to the experiment folder where the plots will be saved.
    """
    ansatz_labels = {
        "HEA_1": "HEA_1 (RY)",
        "HEA_2": "HEA_2 (RYRX)",
        "HEA_3": "HEA_3 (ARY)"
    }

    metrics_df['ansatz_type'] = metrics_df['ansatz_type'].replace(ansatz_labels)
    metrics_df['ansatz_layers'] = metrics_df['ansatz_layers'].astype(str)  # Convert to string
    metrics_df['ansatz_label'] = metrics_df['ansatz_type'] + " - Layers: " + metrics_df['ansatz_layers']

    # Define marker styles for different layers
    marker_styles = {"1": "o", "3": "s", "5": "^"}

    # Define colors for ansatz types
    palette = {"HEA_1 (RY)": "blue", "HEA_2 (RYRX)": "red", "HEA_3 (ARY)": "green"}
    shade_palette = {"HEA_1 (RY)": "lightblue", "HEA_2 (RYRX)": "lightcoral", "HEA_3 (ARY)": "lightgreen"}

    # Define a separate palette for ansatz layers (needed for FacetGrid)
    layer_palette = {"1": "black", "3": "gray", "5": "dimgray"}

    metrics_to_plot = [
        ('mean_expressibility', 'q10_expressibility', 'q90_expressibility', 'Expressibility'),
        ('mean_entangling', 'q10_entangling', 'q90_entangling', 'Entangling Capability')
    ]

    for metric, q10_col, q90_col, metric_label in metrics_to_plot:
        if metric not in metrics_df.columns or metrics_df[metric].isna().all():
            print(f"Skipping {metric_label} because all values are NaN.")
            continue

        # **Single grouped plot with all ansätze together**
        plt.figure(figsize=(10, 6))
        for ansatz in metrics_df['ansatz_type'].unique():
            subset = metrics_df[metrics_df['ansatz_type'] == ansatz]
            if subset[metric].isna().all():
                continue  # Skip missing values

            for layer in subset['ansatz_layers'].unique():
                layer_data = subset[subset['ansatz_layers'] == layer]
                sns.lineplot(
                    data=layer_data,
                    x=x_column,
                    y=metric,
                    label=f"{ansatz} - Layers: {layer}",
                    marker=marker_styles.get(layer, "o"),
                    color=palette.get(ansatz, "black")
                )

                if q10_col in layer_data.columns and q90_col in layer_data.columns:
                    plt.fill_between(layer_data[x_column],
                                     layer_data[q10_col],
                                     layer_data[q90_col], alpha=0.1, color=shade_palette.get(ansatz, "gray"))

        # **Legend customization**
        marker_legend = [Line2D([0], [0], marker=marker_styles[layer], linestyle="", color="black", label=f"Layers: {layer}") for layer in marker_styles]
        color_legend = [Line2D([0], [0], linestyle="-", color=palette[ansatz], label=ansatz) for ansatz in palette]
        legend1 = plt.legend(handles=color_legend, title="Ansatz Type", loc="upper right")
        plt.gca().add_artist(legend1)
        plt.legend(handles=marker_legend, title="Layers", loc="upper left")

        plt.title(f"{metric_label} vs {x_column.replace('_', ' ').title()}")
        plt.xlabel(x_column.replace('_', ' ').title())
        plt.ylabel(metric_label)
        plt.grid(True)

        save_path = os.path.join(experiment_folder, f"{metric}_vs_{x_column}_by_ansatz_type_and_layers.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")

        g = sns.FacetGrid(metrics_df, col="ansatz_type", height=5, aspect=1.2, sharey=True, sharex=True)

        for ax, ansatz in zip(g.axes.flat, metrics_df['ansatz_type'].unique()):
            subset = metrics_df[metrics_df['ansatz_type'] == ansatz]
            for layer in subset['ansatz_layers'].unique():
                layer_data = subset[subset['ansatz_layers'] == layer]
                if layer_data[q10_col].isna().all() or layer_data[q90_col].isna().all():
                    continue

                # Manually set line colors to match ansatz type
                line_color = palette[ansatz]  # Force correct ansatz color
                ax.fill_between(layer_data[x_column],
                                layer_data[q10_col],
                                layer_data[q90_col],
                                alpha=0.1, color=shade_palette.get(ansatz, "gray"))
                ax.plot(layer_data[x_column], layer_data[metric],
                        marker=marker_styles.get(layer, "o"), linestyle="-",
                        color=line_color)  # Force line color

        # **Set axis labels manually**
        for ax in g.axes.flat:
            ax.set_xlabel(x_column.replace("_", " ").title())  # Set x-label for all subplots
        g.axes[0, 0].set_ylabel(metric_label)  # Set y-label only for the leftmost subplot

        # **Fix subplot titles to only show "HEA_1", "HEA_2", etc.**
        for ax, ansatz in zip(g.axes.flat, metrics_df['ansatz_type'].unique()):
            ax.set_title(ansatz.replace("ansatz_type = ", ""))  # Remove unnecessary text

        # **Manually add legend in the rightmost subplot**
        rightmost_ax = g.axes.flat[-1]  # Get the rightmost subplot
        layer_legend = [
            Line2D([0], [0], marker=marker_styles[layer], linestyle="", color="black", label=f"Layers: {layer}")
            for layer in marker_styles
        ]
        rightmost_ax.legend(handles=layer_legend, title="Layers", loc="upper right")

        plt.subplots_adjust(top=0.85)
        g.fig.suptitle(f"{metric_label} vs {x_column.replace('_', ' ').title()}")

        save_path = os.path.join(experiment_folder, f"{metric}_vs_{x_column}_facet_by_ansatz_type.png")
        g.savefig(save_path)
        print(f"Saved plot to {save_path}")


def plot_combined_approximation_ratio(metrics_df, x_column, experiment_folder):
    """
    Plot approximation ratio with error bands, grouped by ansatz type and layers.
    It generates both a single grouped plot and separate subplots per ansatz type.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing the approximation ratio metrics.
        x_column (str): The column name for the x-axis (e.g., 'dropout_ratio', 'param_theta').
        experiment_folder (str): Path to the experiment folder where the plots will be saved.
    """
    ansatz_labels = {
        "HEA_1": "HEA_1 (RY)",
        "HEA_2": "HEA_2 (RYRX)",
        "HEA_3": "HEA_3 (ARY)"
    }

    metrics_df['ansatz_type'] = metrics_df['ansatz_type'].replace(ansatz_labels)
    metrics_df['ansatz_layers'] = metrics_df['ansatz_layers'].astype(str)  # Convert to string
    metrics_df['ansatz_label'] = metrics_df['ansatz_type'] + " - Layers: " + metrics_df['ansatz_layers']

    # Define marker styles for different layers
    marker_styles = {"1": "o", "3": "s", "5": "^"}
    palette = {"HEA_1 (RY)": "blue", "HEA_2 (RYRX)": "red", "HEA_3 (ARY)": "green"}
    shade_palette = {"HEA_1 (RY)": "lightblue", "HEA_2 (RYRX)": "lightcoral", "HEA_3 (ARY)": "lightgreen"}

    metric = 'mean_approximation'
    q10_col = 'q10_approximation'
    q90_col = 'q90_approximation'
    metric_label = 'Approximation Ratio'

    if metric not in metrics_df.columns or metrics_df[metric].isna().all():
        print(f"Skipping {metric_label} because all values are NaN.")
        return

    # **Single grouped plot with variability bands**
    plt.figure(figsize=(10, 6))

    legend_handles = []
    for ansatz in metrics_df['ansatz_type'].unique():
        subset = metrics_df[metrics_df['ansatz_type'] == ansatz]
        if subset[metric].isna().all():
            continue  # Skip missing values

        for layer in subset['ansatz_layers'].unique():
            layer_data = subset[subset['ansatz_layers'] == layer]
            line, = plt.plot(layer_data[x_column], layer_data[metric],
                             marker=marker_styles.get(layer, "o"),
                             color=palette.get(ansatz, "black"),
                             linestyle="-")
            legend_handles.append(line)

            if q10_col in layer_data.columns and q90_col in layer_data.columns:
                plt.fill_between(layer_data[x_column],
                                 layer_data[q10_col],
                                 layer_data[q90_col], alpha=0.1, color=shade_palette.get(ansatz, "gray"))

    # Create separate legends for Ansatz types and Layers
    marker_legend = [Line2D([0], [0], marker=marker_styles[layer], linestyle="", color="black", label=f"Layers: {layer}") for layer in marker_styles]
    color_legend = [Line2D([0], [0], linestyle="-", color=palette[ansatz], label=ansatz) for ansatz in palette]
    legend1 = plt.legend(handles=color_legend, title="Ansatz Type", loc="upper right")
    plt.gca().add_artist(legend1)
    plt.legend(handles=marker_legend, title="Layers", loc="upper left")

    plt.title(f"{metric_label} vs {x_column.replace('_', ' ').title()}")
    plt.xlabel(x_column.replace('_', ' ').title())
    plt.ylabel(metric_label)
    plt.grid(True)

    save_path = os.path.join(experiment_folder, f"approximation_ratio_vs_{x_column}_by_ansatz_type_and_layers.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

    # **FacetGrid for separate subplots per ansatz_type with Q10-Q90 shadow and different markers**
    g = sns.FacetGrid(metrics_df, col="ansatz_type", height=5, aspect=1.2, sharey=True, sharex=True)

    for ax, ansatz in zip(g.axes.flat, metrics_df['ansatz_type'].unique()):
        subset = metrics_df[metrics_df['ansatz_type'] == ansatz]
        for layer in subset['ansatz_layers'].unique():
            layer_data = subset[subset['ansatz_layers'] == layer]
            if layer_data[q10_col].isna().all() or layer_data[q90_col].isna().all():
                continue

            line_color = palette[ansatz]  # Ensure correct ansatz color
            # ax.fill_between(layer_data[x_column],
            #                 layer_data[q10_col],
            #                 layer_data[q90_col],
            #                 alpha=0.1, color=shade_palette.get(ansatz, "gray"))
            ax.plot(layer_data[x_column], layer_data[metric],
                    marker=marker_styles.get(layer, "o"), linestyle="-",
                    color=line_color)  # Force line color

    # **Set axis labels manually**
    for ax in g.axes.flat:
        ax.set_xlabel(x_column.replace("_", " ").title())  # X-label for all subplots
    g.axes[0, 0].set_ylabel(metric_label)  # Y-label only for leftmost subplot

    # **Fix subplot titles to only show "HEA_1", "HEA_2", "HEA_3"**
    for ax, ansatz in zip(g.axes.flat, metrics_df['ansatz_type'].unique()):
        ax.set_title(ansatz.replace("ansatz_type = ", ""))  # Clean title

    # **Manually add legend in the rightmost subplot**
    rightmost_ax = g.axes.flat[-1]  # Get the rightmost subplot
    layer_legend = [
        Line2D([0], [0], marker=marker_styles[layer], linestyle="", color="black", label=f"Layers: {layer}")
        for layer in marker_styles
    ]
    rightmost_ax.legend(handles=layer_legend, title="Layers", loc="upper right")

    plt.subplots_adjust(top=0.85)
    g.fig.suptitle(f"{metric_label} vs {x_column.replace('_', ' ').title()}")

    save_path = os.path.join(experiment_folder, f"approximation_ratio_vs_{x_column}_facet_by_ansatz_type.png")
    g.savefig(save_path)
    print(f"Saved plot to {save_path}")