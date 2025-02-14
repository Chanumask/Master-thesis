import numpy as np
import os
from qiskit import transpile
from qiskit.primitives import Sampler
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit.quantum_info import DensityMatrix, partial_trace
from qiskit import QuantumCircuit

from scipy.special import rel_entr
from scipy.stats import entropy as scipy_entropy
from matplotlib import pyplot as plt

from plotting import plot_expressibility_entanglement
from util import generate_qpic_circuit


def run_circuit(circuit, shots=1024):
    """Simulate the circuit and return the measurement counts."""
    if circuit.num_clbits == 0:
        # Add measurement gates if not present
        circuit = circuit.copy()
        circuit.measure_all()

    simulator = AerSimulator()
    transpiled_circuit = transpile(circuit, simulator)  # Pre-transpile once
    simulator.set_options(seed_simulator=1)
    result = simulator.run(transpiled_circuit, shots=shots).result()
    return result.get_counts()


def simulate_statevector(circuit: QuantumCircuit):
    """Simulates the circuit using Aers StateVectorSimulator and returns the results statevector"""
    simulator = StatevectorSimulator()
    transpiled_circuit = transpile(circuit, simulator)
    result = simulator.run(transpiled_circuit).result()
    statevector = result.get_statevector()
    return statevector


def run_sampler_on_circuit(circuit, shots=1024):
    """Run the sampler, obtain the solution distribution and get the most probable solution."""
    if circuit.num_clbits == 0:
        circuit = circuit.copy()
        circuit.measure_all()
    sampler = Sampler(options={"shots": shots})
    result = sampler.run([circuit]).result()
    return result


def calculate_cut_cost(graph, solution):
    """Calculate the cost (number of cuts) for the Max-Cut problem based on the solution."""
    cut_value = 0
    for u, v, weight in graph.edges(data="weight", default=1):  # assuming graph edges have weights
        # Check if the endpoints of the edge (u, v) are in different sets
        if solution[u] != solution[v]:
            cut_value += weight
    return cut_value


def get_probabilities(circuit, shots=1024):
    """
    Obtain the probability distribution from sampling the circuit.

    Args:
        circuit (QuantumCircuit): The quantum circuit to sample.
        shots (int): Number of shots for sampling.

    Returns:
        dict: A dictionary of bitstrings to probabilities.
    """
    result = run_sampler_on_circuit(circuit, shots=shots)
    quasi_dists = result.quasi_dists[0]
    probs = {format(key, f'0{circuit.num_qubits}b'): float(value) for key, value in quasi_dists.items()}

    # Normalize probabilities if necessary
    total_prob = sum(probs.values())
    if not np.isclose(total_prob, 1.0):
        probs = {k: v / total_prob for k, v in probs.items()}

    return probs

def calculate_ansatz_metrics(ansatz, config):
    """If specified in the config this calculates expressibility and entangling capability as well as generating a qpic file to illustrate the ansatz"""
    directory = "ansätze"
    ansatz_type = config['vqe_parameters']['ansatz_type']
    ansatz_layers = config['vqe_parameters']['ansatz_layers']
    entangle_method = config['vqe_parameters']['entangle_method']
    expressibility_score = None
    entangling_capability_score = None

    if config['eval']['ent_cap']:
        entangling_capability_score = entanglement_capability(ansatz)

    if config['eval']['qpic']:
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Generate QPIC filename
        qpic_filename = f"{directory}/{ansatz_type}.qpic"

        # Generate QPIC file
        if not os.path.exists(qpic_filename):
            qpic_str = generate_qpic_circuit(ansatz)
            with open(qpic_filename, 'w', encoding='utf-8') as file:
                file.write(qpic_str)
            print(f"Generated QPIC file for ansatz: {qpic_filename}")
        else:
            print(f"QPIC file already exists: {qpic_filename}")

    if config['eval']['expressibility']:
        # Determine the suffix based on the entangle method
        if entangle_method == 'dropout':
            suffix = f"d{config['vqe_parameters']['dropout_ratio']}"
        elif entangle_method == 'parametrize':
            param_theta = config['vqe_parameters']['param_theta']

            if 'pi' in param_theta:
                # Extract everything after 'pi' (e.g., "/2" or "/4") and replace "/" with "_"
                pi_part = param_theta.split('pi')[1].strip()  # Gets the part after 'pi'
                sanitized_theta = f"pi{pi_part.replace('/', '_')}" if pi_part else "pi"
            else:
                # If 'pi' is not in the string, keep the original value as the suffix
                sanitized_theta = param_theta.replace("/", "_")
            suffix = f"p{sanitized_theta}"

        else:
            suffix = ""

        # Construct the expressibility filename
        expressibility_filename = f"{directory}/{ansatz_type}_{ansatz_layers}_{suffix}".strip("_")

        # Check if the file already exists
        if not os.path.exists(os.path.join(expressibility_filename, "_expressibility.png")):
            expressibility_score = compute_expressibility(ansatz, expressibility_filename,
                                                          plot_expressibility=config['eval']['expr_plot'])
            print(f"Generated expressibility graph: {expressibility_filename}")
        else:
            print(f"Expressibility graph already exists: {expressibility_filename}")

    return {"expressibility": expressibility_score, "entangling_capability": entangling_capability_score}


def entanglement_capability(ansatz, num_samples=1000, measure="meyer-wallach"):
    """
    Calculate entanglement capability of a circuit.

    Args:
        ansatz (QuantumCircuit): The parametrized quantum circuit (with no bound parameters).
        num_samples (int): Number of samples for the experiment.
        measure (str): Measure for entanglement capability, e.g., "meyer-wallach".

    Returns:
        float: Entanglement capability score.
    """
    if ansatz.num_parameters == 0:
        raise ValueError("Ansatz has no parameters to sample for entanglement capability calculation.")

    num_qubits = ansatz.num_qubits

    bounds = getattr(ansatz, "parameter_bounds", None)
    if bounds is None:
        bounds = [(-2 * np.pi, 2 * np.pi)] * ansatz.num_parameters

    lower_bounds, upper_bounds = zip(*[(low if low is not None else -2 * np.pi,
                                        high if high is not None else 2 * np.pi)
                                       for low, high in bounds])

    # Generate `num_samples` full sets of parameters
    param_list = [np.random.uniform(low=lower_bounds, high=upper_bounds, size=ansatz.num_parameters)
                  for _ in range(num_samples)]

    # param_list = [2 * np.pi * np.random.rand(ansatz.num_parameters) for _ in range(num_samples)]
    # print(param_list)

    entanglement_scores = []

    for params in param_list:
        # Assign parameters to the circuit
        circuit = ansatz.assign_parameters(params)
        counts = run_circuit(circuit, shots=num_samples)

        total_shots = sum(counts.values())
        sum_trace_rho_squared = 0

        for i in range(num_qubits):
            # Marginalize probabilities for the i-th qubit
            marginal_probs = {}
            for bitstring, count in counts.items():
                key = bitstring[-(i + 1)]  # Extract the i-th qubit measurement result
                marginal_probs[key] = marginal_probs.get(key, 0) + count

            # Normalize probabilities
            marginal_probs = {key: value / total_shots for key, value in marginal_probs.items()}

            # Compute Tr(rho_i^2) for the i-th qubit
            trace_rho_squared = sum(prob ** 2 for prob in marginal_probs.values())
            sum_trace_rho_squared += trace_rho_squared

        q = (4 / num_qubits) * (1 - (sum_trace_rho_squared/num_qubits))  # Meyer-Wallach measure
        entanglement_scores.append(q)

    avg_entanglement = np.mean(entanglement_scores)
    print(f"entanglement capability: {avg_entanglement}")

    return avg_entanglement

def compute_expressibility(ansatz, ansatz_name, num_samples=1000, bins=50, plot_expressibility=False):
    """
    Calculate the expressibility of a given ansatz by comparing the fidelity distribution
    to the Haar distribution.
    Args:
        ansatz_name: from the config used for saving the plot.
        ansatz (QuantumCircuit): A parameterized quantum circuit (with no bound parameters).
        num_samples (int): Number of random parameter configurations for the ansatz.
        bins (int): Number of bins for histogram. Default is 50.
        plot_expressibility (bool): should be false when working with slurm.

    Returns:
        float: KL divergence as the expressibility metric.
    """
    if ansatz.num_parameters == 0:
        raise ValueError("Ansatz has no parameters to bind for expressibility calculation.")

    num_qubits = ansatz.num_qubits


    bounds = getattr(ansatz, "parameter_bounds", None)
    if bounds is None:
        bounds = [(-2 * np.pi, 2 * np.pi)] * ansatz.num_parameters

    lower_bounds, upper_bounds = zip(*[(low if low is not None else -2 * np.pi,
                                        high if high is not None else 2 * np.pi)
                                       for low, high in bounds])

    # Generate `num_samples` full sets of parameters
    param_list = [np.random.uniform(low=lower_bounds, high=upper_bounds, size=ansatz.num_parameters)
                  for _ in range(num_samples)]

    # param_list = [2 * np.pi * np.random.rand(ansatz.num_parameters) for _ in range(num_samples)]
    #print(param_list)

    # Step 1: Generate fidelities for the ansatz
    circuit_fidelities = []

    for params_1 in param_list:
        circuit_1 = ansatz.assign_parameters(params_1)
        counts_1 = run_circuit(circuit_1)

        # Generate a random second set of parameters
        params_2 = 2 * np.pi * np.random.rand(ansatz.num_parameters)
        circuit_2 = ansatz.assign_parameters(params_2)
        counts_2 = run_circuit(circuit_2)

        # Compute fidelity between the two distributions+
        fidelity = calculate_fidelity_from_counts(counts_1, counts_2, num_samples)
        circuit_fidelities.append(fidelity)

    # Step 2: Generate Haar-random fidelities
    fidelity_range = np.linspace(0, 1, bins)
    haar_distribution = (2 ** num_qubits - 1) * (1 - fidelity_range + 1e-8) ** (2 ** num_qubits - 2)
    haar_prob = haar_distribution / haar_distribution.sum()
    haar_fidelities = np.random.rand(num_samples)
    haar_prob, _ = np.histogram(haar_fidelities, bins=bins, density=True)
    haar_prob /= np.sum(haar_prob)

    # Step 3: Compute histograms
    circuit_hist, bin_edges = np.histogram(circuit_fidelities, bins=bins, range=(0, 1), density=True)
    circuit_prob = circuit_hist / circuit_hist.sum()
    kl_divergence = np.sum(rel_entr(circuit_prob + 1e-12, haar_prob + 1e-12))

    if plot_expressibility:
        bin_middles = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        plt.bar(bin_middles, circuit_prob, width=(1 / bins), alpha=0.5, label="Circuit")

        plt.plot(fidelity_range, haar_prob, label="Haar", color="orange")

        plt.legend()
        plt.title(f"Expressibility ({ansatz_name}), KL Divergence: {kl_divergence:.3f})")
        plt.xlabel("Fidelity")
        plt.ylabel("Probability")

        directory = "ansätze"
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f"{ansatz_name}_expressibility.png")
        plt.close()
        # print(f"KL Divergence: {kl_divergence}")
    return kl_divergence

def calculate_fidelity_from_counts(counts_1, counts_2, shots):
    """
    Calculate fidelity between two measurement distributions (counts).

    Args:
        counts_1 (dict): Measurement results from the first circuit.
        counts_2 (dict): Measurement results from the second circuit.
        shots (int): Number of shots in each measurement.

    Returns:
        float: Fidelity between the two distributions.
    """
    # Convert counts to probabilities
    probs_1 = {key: value / shots for key, value in counts_1.items()}
    probs_2 = {key: value / shots for key, value in counts_2.items()}

    # Calculate fidelity
    keys = set(probs_1.keys()).union(probs_2.keys())
    fidelity = sum(np.sqrt(probs_1.get(key, 0) * probs_2.get(key, 0)) for key in keys)

    return fidelity

# def calculate_entanglement_metrics(circuit, probabilities):
#     """
#     Calculate entanglement metrics of the optimized circuit using the given probabilities.
#
#     Args:
#         circuit (QuantumCircuit): The quantum circuit analyzed.
#         probabilities (dict): Probability distribution of measured bit strings
#
#     Returns:
#         dict: A dictionary containing the entanglement metrics.
#     """
#     num_qubits = circuit.num_qubits
#     dim = 2 ** num_qubits
#
#     # Build density matrix from probabilities
#     density_matrix = np.zeros((dim, dim), dtype=complex)
#     for bitstring, prob in probabilities.items():
#         index = int(bitstring, 2)
#         density_matrix[index, index] = prob
#     rho = DensityMatrix(density_matrix)
#
#     # Check if it's a valid probability distribution
#     diag_probs = np.diag(rho.data).real
#     if not np.isclose(np.sum(diag_probs), 1):
#         raise ValueError("Invalid probability distribution")
#
#     # Entanlement per Qubit (von Neumann Entropy)
#     # https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.3.040339
#     # https://journals.aps.org/pra/pdf/10.1103/PhysRevA.109.032413
#     entropies = []
#     for i in range(num_qubits):
#         # Partial trace over the other qubits to get the reduced density matrix for qubit i
#         partial_rho = partial_trace(rho, [j for j in range(num_qubits) if j != i])
#         partial_probabilities = np.diag(partial_rho.data).real
#         entropies.append(scipy_entropy(partial_probabilities))
#     average_entanglement_per_qubit = np.mean(entropies)
#
#     # Meyer-Wallach Measure  http://arxiv.org/abs/1905.10876
#     traces = []
#     for i in range(num_qubits):
#         # Partial trace over other qubits to get rho_k
#         rest = [j for j in range(num_qubits) if j != i]
#         rho_k = partial_trace(rho, rest)
#         # Compute Tr[rho_k^2]
#         rho_k_matrix = rho_k.data
#         trace_rho_k_squared = np.trace(rho_k_matrix @ rho_k_matrix)
#         traces.append(trace_rho_k_squared.real)
#
#     # Compute Q
#     average_trace = np.mean(traces)
#     meyer_wallach = 2 * (1 - average_trace)
#
#     metrics = {
#         'e_per_qubit': average_entanglement_per_qubit,
#         'meyer_wallach': meyer_wallach
#     }
#
#     return metrics