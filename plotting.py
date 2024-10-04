import ast
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
