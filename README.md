# VQE Entanglement Impact

This repository contains a quantum computing project that implements the **Variational Quantum Eigensolver (VQE)** to solve the **MaxCut problem** on quantum hardware simulators. The focus is on exploring different **entanglement strategies** and analyzing their impact on the performance of the VQE.

## Main Features

- **Configurable Experiments:**  
  All parameters and settings are defined in a **configuration file**, allowing for flexible adjustments.

- **Experiment Execution & Visualization:**  
  - To **run an experiment**, use:  
    ```python
    experiment_path = run_experiment(...)
    ```
    This generates an **experiment folder** inside the `results` directory.  
  - To **visualize results**, pass either the generated path or a pre-existing one (if data is already available):  
    ```python
    experiment_path = "results"  # Example for existing data
    ```
    This path is then used in reading and plotting functions to compute desired metrics.

- **Circuit Visualization:**  
  - Generate images of quantum circuits that have been preprocessed using `util.generate_qpic_circuit`:  
    ```sh
    python .venv/Scripts/qpic -f pdf circuit.qpic
    ```

## Experiment Workflow

### Parallelized Experiment Execution

To improve parallelization, experiments are split into separate runs:
- **Approximation ratio** experiments are executed independently from those evaluating **expressibility** and **entangling capability**.
- The **approximation ratio** experiments use **50 graphs** (`config:iterations`).
- **Expressibility and entangling capability** experiments use **only one graph**, as their sampling numbers are managed separately from `config:iterations`.
- These experiments can still be combined within a single run since **expressibility and entangling capability are always computed during the first iteration**.

### Running Experiments

1. **Execute `run_experiment`** to generate an experiment folder inside the `results` directory.
2. **Run configurations**:
   - If a parameter is provided as a list in the configuration, it is treated as a range, and **a separate run is executed for each value**.
3. **Access Results**:
   - Each **run folder** contains a **CSV file** with recorded data.
   - The path containing the seed folders can then be used to read and plot desired metrics, which are **averaged across runs** to provide the final results.

---

