import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit.primitives import Sampler
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit.quantum_info import DensityMatrix, partial_trace, entropy, concurrence, purity
from qiskit import QuantumCircuit


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


def plot_solution(dist, sol, filename):
    """Plots histogram and prints most probable solution."""
    # Check if 'dist' indeed includes all states, if not, the previous method will be necessary
    # Here we assume it includes all states

    # Extracting probability data
    probabilities = dist.binary_probabilities()

    # Plotting
    fig, ax = plt.subplots()
    ax.bar(probabilities.keys(), probabilities.values(), color='blue')
    ax.set_xlabel('State')
    ax.set_ylabel('Probability')
    ax.set_title('Probability Distribution of Quantum States')
    plt.xticks(rotation=90)  # Rotate state labels for better visibility
    plt.tight_layout()

    # Save the figure
    plt.savefig(filename)
    plt.show()

    # Print most probable solution
    print(f"State: {sol[0]}, Probability: {sol[1]}")
    for i, group in enumerate(reversed(sol[0])):
        print(f"Node {i}: {group}")


def calculate_entanglement_metrics(circuit: QuantumCircuit):
    """Calculate various entanglement metrics for an (optimized) quantum circuit."""
    # Simulate the statevector of the optimized circuit
    statevector = simulate_statevector(circuit)

    # Create a density matrix from the statevector
    rho = DensityMatrix(statevector)

    # Calculate metrics
    metrics = {}
    num_qubits = circuit.num_qubits

    # 1. Entanglement Entropy
    metrics['entanglement_entropy'] = entropy(rho)

    # 2. Concurrence for the first two qubits
    if num_qubits >= 2:
        partial_rho_2q = partial_trace(rho, list(range(2, num_qubits)))
        metrics['concurrence'] = concurrence(partial_rho_2q)

    # 3. Purity
    metrics['purity'] = purity(rho)

    return metrics
