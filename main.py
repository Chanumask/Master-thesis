import numpy as np

from util import load_config, run_experiment

if __name__ == '__main__':
    np.random.seed(1)

    config = load_config('conf_files/max_cut/fully_connected.yaml')

    run_experiment(config)

    # evaluate_csv("vqe_results.csv")
