# VQE Entanglement Impact

This repository implements the **Variational Quantum Eigensolver (VQE)** to solve the **MaxCut problem** on quantum hardware simulators. The focus is on exploring different **entanglement strategies** and analyzing their impact on VQE performance.

## Setup

1. **Install dependencies** using:
   ```sh
   pip install -r requirements.txt
   ```
2. Run experiments using the provided scripts (see below for details).

## Running Experiments

Experiments are configured via a config file, where parameters can be adjusted. Running:
    ```python
    experiment_path = run_experiment(...)
    ```
    This generates an **experiment folder** inside the `results` directory.  
  - To **visualize results**, pass either the generated path or a pre-existing one (if data is already available).
    This path is then used in reading and plotting functions to compute desired metrics.

- **Circuit Visualization:**  
  - Generate images of quantum circuits that have been preprocessed using `util.generate_qpic_circuit`:  
    ```sh
    python .venv/Scripts/qpic -f pdf circuit.qpic
    ```
## Parallelization Strategy

To improve parallelization, experiments are split into separate runs:
- **Approximation ratio** experiments are executed independently from those evaluating **expressibility** and **entangling capability**.
- The **approximation ratio** experiments use **50 graphs** (`config:iterations`).
- **Expressibility and entangling capability** experiments use **only one graph**, as their sampling numbers are managed separately from `config:iterations`.
- These experiments can still be combined within a single run since **expressibility and entangling capability are always computed during the first iteration**.
