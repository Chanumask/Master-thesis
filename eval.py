import numpy as np
from qiskit import transpile
from qiskit.primitives import Sampler
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit.quantum_info import DensityMatrix, partial_trace, entropy, concurrence, state_fidelity
from qiskit import QuantumCircuit

from scipy.stats import entropy as scipy_entropy

def run_circuit(circuit):
    simulator = AerSimulator()
    simulator.set_options(seed_simulator=1)
    result = simulator.run(transpile(circuit, simulator), shots=1024).result()
    counts = result.get_counts(circuit)
    return counts


def simulate_statevector(circuit: QuantumCircuit):
    """Simulates the circuit using Aers StateVectorSimulator and returns the results statevector"""
    simulator = StatevectorSimulator()
    transpiled_circuit = transpile(circuit, simulator)
    result = simulator.run(transpiled_circuit).result()
    statevector = result.get_statevector()
    return statevector


def run_sampler_on_circuit(circuit):
    """Run the sampler, obtain the solution distribution and get the most probable solution."""
    if circuit.num_clbits == 0:
        circuit = circuit.copy()
        circuit.measure_all()
    sampler = Sampler(options={"shots": 1024})
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


def get_probabilities(circuit, num_shots=1024):
    """
    Obtain the probability distribution from sampling the circuit.

    Args:
        circuit (QuantumCircuit): The quantum circuit to sample.
        num_shots (int): Number of shots for sampling.

    Returns:
        dict: A dictionary of bitstrings to probabilities.
    """
    result = run_sampler_on_circuit(circuit)
    quasi_dists = result.quasi_dists[0]
    probs = {format(key, f'0{circuit.num_qubits}b'): float(value) for key, value in quasi_dists.items()}

    # Normalize probabilities if necessary
    total_prob = sum(probs.values())
    if not np.isclose(total_prob, 1.0):
        probs = {k: v / total_prob for k, v in probs.items()}

    return probs


def calculate_entanglement_metrics(circuit, probabilities):
    """
    Calculate entanglement metrics using the given probabilities.

    Args:
        circuit (QuantumCircuit): The quantum circuit analyzed.
        probabilities (dict): Probability distribution of measured bit strings

    Returns:
        dict: A dictionary containing the entanglement metrics.
    """
    num_qubits = circuit.num_qubits
    dim = 2 ** num_qubits

    # Build density matrix from probabilities
    density_matrix = np.zeros((dim, dim), dtype=complex)
    for bitstring, prob in probabilities.items():
        index = int(bitstring, 2)
        density_matrix[index, index] = prob
    rho = DensityMatrix(density_matrix)

    # Check if it's a valid probability distribution
    diag_probs = np.diag(rho.data).real
    if not np.isclose(np.sum(diag_probs), 1):
        raise ValueError("Invalid probability distribution")

    # Entanlement per Qubit (von Neumann Entropy)
    # https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.3.040339
    # https://journals.aps.org/pra/pdf/10.1103/PhysRevA.109.032413
    entropies = []
    for i in range(num_qubits):
        # Partial trace over the other qubits to get the reduced density matrix for qubit i
        partial_rho = partial_trace(rho, [j for j in range(num_qubits) if j != i])
        partial_probabilities = np.diag(partial_rho.data).real
        entropies.append(scipy_entropy(partial_probabilities))
    average_entanglement_per_qubit = np.mean(entropies)

    # Meyer-Wallach Measure  http://arxiv.org/abs/1905.10876
    traces = []
    for i in range(num_qubits):
        # Partial trace over other qubits to get rho_k
        rest = [j for j in range(num_qubits) if j != i]
        rho_k = partial_trace(rho, rest)
        # Compute Tr[rho_k^2]
        rho_k_matrix = rho_k.data
        trace_rho_k_squared = np.trace(rho_k_matrix @ rho_k_matrix)
        traces.append(trace_rho_k_squared.real)

    # Compute Q
    average_trace = np.mean(traces)
    meyer_wallach = 2 * (1 - average_trace)

    metrics = {
        'e_per_qubit': average_entanglement_per_qubit,
        'meyer_wallach': meyer_wallach
    }

    return probabilities, metrics


# def mw_measure(circuit, num_qubits, sample=1024):
#     """
#     Returns the meyer-wallach entanglement measure for the given circuit.
#     """
#
#     temp = np.zeros(sample, dtype=complex)
#     N = num_qubits
#
#     for i in range(sample):
#         statevector = simulate_statevector(circuit)
#         entropy_mw = 0
#         qb_list = list(range(N))
#
#         # rho_k after tracing out the rest
#         for j in range(N):
#             rest = qb_list[:j] + qb_list[j + 1:]
#             dens = partial_trace(statevector, rest).data
#             trace = np.trace(np.matmul(dens, dens))  # rho_k^2
#             entropy_mw += trace
#
#         entropy_mw /= N
#         temp[i] = 1.0 - entropy_mw
#
#     # Meyer-Wallach measure
#     Q = 2 * np.sum(temp).real / sample
#
#     return Q


def calculate_expressibility(ansatz:QuantumCircuit, num_samples=100, num_qubits=4, num_fidelity_bins=50):
    """
    Calculate the expressibility of a given ansatz by comparing the fidelity distribution
    to the Haar distribution.

    Args:
        ansatz (QuantumCircuit): Parameterized quantum circuit representing the ansatz.
        num_samples (int): Number of random parameter sets to sample.
        num_qubits (int): Number of qubits in the circuit.
        num_fidelity_bins (int): Number of bins for the fidelity histogram.

    Returns:
        float: KL divergence representing the expressibility of the ansatz.
    """

    if ansatz.num_parameters == 0:
        raise ValueError("Ansatz has no parameters to bind for expressibility calculation.")

    # Generate random parameter sets
    num_qubits = ansatz.num_qubits
    fidelities = []

    for _ in range(num_samples):
        # Generate two random parameter sets
        param_values_1 = 2 * np.pi * np.random.rand(num_qubits)
        param_values_2 = 2 * np.pi * np.random.rand(num_qubits)

        # Bind parameters to create two different instances of the ansatz
        circ1 = ansatz.assign_parameters(param_values_1)
        circ2 = ansatz.assign_parameters(param_values_2)

        # Get statevectors using simulate_statevector
        statevector1 = simulate_statevector(circ1)
        statevector2 = simulate_statevector(circ2)

        # Calculate the fidelity between the two statevectors
        fidelity = state_fidelity(statevector1, statevector2)
        fidelities.append(fidelity)

    # Create a normalized histogram of fidelities
    hist, bin_edges = np.histogram(fidelities, bins=num_fidelity_bins, range=(0, 1), density=True)
    hist = hist / np.sum(hist)

    # Approximate Haar distribution (uniform for simplicity)
    haar_dist = np.ones(num_fidelity_bins) / num_fidelity_bins

    # Calculate KL divergence as the expressibility metric
    kl_divergence = scipy_entropy(hist, haar_dist)
    return kl_divergence
