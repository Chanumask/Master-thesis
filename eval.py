import pandas as pd


def evaluate_csv(filepath):
    data = pd.read_csv(filepath)

    average_optimal_values = data.groupby('n_entanglements')['optimal_value'].mean()
    sample_counts = data['n_entanglements'].value_counts().sort_index()

    print("Average Optimal Value per Number of Entanglements:")
    print(average_optimal_values)
    print("\nSample Count per Number of Entanglements:")
    print(sample_counts)