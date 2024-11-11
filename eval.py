import numpy as np
from qiskit import transpile
from qiskit.primitives import Sampler
from qiskit_aer import AerSimulator, StatevectorSimulator, Aer
from qiskit.quantum_info import DensityMatrix, partial_trace, entropy, concurrence, purity, state_fidelity
from qiskit import QuantumCircuit

from scipy.stats import entropy
from qiskit.circuit import ParameterVector

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


def get_solution(optimal_state):
    """Run the sampler, obtain the solution distribution and get the most probable solution."""
    optimal_state.measure_all()
    sampler = Sampler(options={"shots": 1024})
    result = sampler.run([optimal_state]).result()

    distribution = {format(key, f'0{optimal_state.num_qubits}b'): prob for key, prob in result.quasi_dists[0].items()}
    solution = max(distribution.items(), key=lambda x: x[1])
    return distribution, solution


def calculate_cut_cost(graph, solution):
    """Calculate the cost (number of cuts) for the Max-Cut problem based on the solution."""
    cut_value = 0
    for u, v, weight in graph.edges(data="weight", default=1):  # assuming graph edges have weights
        # Check if the endpoints of the edge (u, v) are in different sets
        if solution[u] != solution[v]:
            cut_value += weight
    return cut_value


def calculate_entanglement_metrics(optimized_circuit: QuantumCircuit):
    """Calculate various entanglement metrics for an (optimized) quantum circuit."""
    # Simulate the statevector of the optimized circuit
    statevector = simulate_statevector(optimized_circuit)

    # Create a density matrix from the statevector
    rho = DensityMatrix(statevector)

    # Calculate metrics
    metrics = {}
    num_qubits = optimized_circuit.num_qubits

    probabilities = np.diag(rho.data).real

    # Check if it's a valid probability distribution (i.e., sums to 1)
    if not np.isclose(np.sum(probabilities), 1):
        raise ValueError("Invalid probability distribution")

    # 1. Entanglement Entropy
    metrics['entanglement_entropy'] = entropy(probabilities)

    # 2. Concurrence for the first two qubits
    if num_qubits >= 2:
        partial_rho_2q = partial_trace(rho, list(range(2, num_qubits)))
        metrics['concurrence'] = concurrence(partial_rho_2q)

    # 3. Purity
    metrics['purity'] = purity(rho)

    # 4. Entanglement per qubit (von Neumann entropy)
    # https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.3.040339
    # https: // journals.aps.org / pra / pdf / 10.1103 / PhysRevA.109.032413
    # e_per_qubit = (2 * np.log2(chi)) / num_qubits
    entropies = []
    for i in range(num_qubits):
        # Partial trace over the other qubits to get the reduced density matrix for qubit i
        partial_rho = partial_trace(rho, [j for j in range(num_qubits) if j != i])
        partial_probabilities = np.diag(partial_rho.data).real

        # Calculate the von Neumann entropy for qubit i
        entropies.append(entropy(partial_probabilities))

    average_entanglement_per_qubit = np.mean(entropies)
    metrics['e_per_qubit'] = average_entanglement_per_qubit

    # 5 averaged Meyer-Wallach entanglement measure  http://arxiv.org/abs/1905.10876
    mw = mw_measure(optimized_circuit, num_qubits)
    metrics['meyer_wallach'] = mw

    return metrics


def mw_measure(circuit, num_qubits, sample=1024):
    """
    Returns the meyer-wallach entanglement measure for the given circuit.
    """

    temp = np.zeros(sample, dtype=complex)
    N = num_qubits

    for i in range(sample):
        statevector = simulate_statevector(circuit)
        entropy_mw = 0
        qb_list = list(range(N))

        # rho_k after tracing out the rest
        for j in range(N):
            rest = qb_list[:j] + qb_list[j + 1:]
            dens = partial_trace(statevector, rest).data
            trace = np.trace(np.matmul(dens, dens))  # rho_k^2
            entropy_mw += trace

        entropy_mw /= N
        temp[i] = 1.0 - entropy_mw

    # Meyer-Wallach measure
    Q = 2 * np.sum(temp).real / sample

    return Q


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
    parameters = ParameterVector('theta', ansatz.num_parameters)
    fidelities = []

    for _ in range(num_samples):
        # Generate two random parameter sets
        param_values_1 = 2 * np.pi * np.random.rand(len(parameters))
        param_values_2 = 2 * np.pi * np.random.rand(len(parameters))

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
    hist, _ = np.histogram(fidelities, bins=num_fidelity_bins, range=(0, 1), density=True)
    hist = hist / np.sum(hist)

    # Approximate Haar distribution (uniform for simplicity)
    haar_dist = np.ones(num_fidelity_bins) / num_fidelity_bins

    # Calculate KL divergence as the expressibility metric
    kl_divergence = entropy(hist, haar_dist)
    return kl_divergence
