import ast
import csv
import os

import pandas as pd
from matplotlib import pyplot as plt


def extract_metrics(metrics_str):
    """Convert the entanglement_metrics string into a dictionary."""
    try:
        metrics_dict = ast.literal_eval(metrics_str)  # Safely parse the string into a dictionary
        return metrics_dict
    except (ValueError, SyntaxError):
        return {}


def plot_metrics_from_csv(relevant_metrics, results_folder):
    """Read CSV files from the results folder and plot relevant metrics."""
    metrics_to_plot = ['entanglement_entropy', 'concurrence', 'purity', 'e_per_qubit', 'meyer_wallach']
    all_data = {metric: [] for metric in metrics_to_plot}
    filenames = []

    for filename in os.listdir(results_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(results_folder, filename)
            filenames.append(filename)

            data = pd.read_csv(filepath)
            # Extract entanglement metrics from each row
            for _, row in data.iterrows():
                metrics_dict = extract_metrics(row['entanglement_metrics'])

                for metric in metrics_to_plot:
                    if metric in metrics_dict:
                        all_data[metric].append(metrics_dict[metric])
                    else:
                        all_data[metric].append(None)  # handle missing values

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
            plt.show()


def plot_energy_history_from_csv(filepath):
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
        plt.savefig("results/energy_history_plot.png", format='png')
