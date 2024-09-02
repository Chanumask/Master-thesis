import yaml

from max_cut import generate_graph_from_config, max_cut_given_graph


def load_config(file_path):
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def run_experiment(config):
    """Run the VQE experiment based on the configuration."""
    problem_class = config['problem_class']

    if problem_class == 'max_cut':
        graph = generate_graph_from_config(config)
        ansatz_layers = config['vqe_parameters']['ansatz_layers']
        entangle_ratio = config['vqe_parameters']['entangle_ratio']
        iterations = config['vqe_parameters']['iterations']
        filepath = config['vqe_parameters']['filepath']

        for i in range(iterations):
            print(f"Running iteration {i + 1}/{iterations}")
            max_cut_given_graph(graph, ansatz_layers, entangle_ratio, filepath)

    else:
        raise ValueError("Unsupported problem class in configuration.")
