import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.primitives import Sampler
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt


def run_circuit(circuit):
    simulator = AerSimulator()
    simulator.set_options(seed_simulator=1)
    result = simulator.run(transpile(circuit, simulator), shots=1024).result()
    counts = result.get_counts(circuit)
    return counts


def get_solution(optimal_state):
    """Run the sampler, obtain the solution distribution and get the most probable solution."""
    sampler = Sampler(options={"shots": 1024})
    distribution = sampler.run([optimal_state]).result().quasi_dists[0]
    solution = max(distribution.binary_probabilities().items(), key=lambda x: x[1])
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


def add_cyclic_entanglement(circuit, entangle_ratio):
    """Add cyclic entanglement using CX gates at the beginning of the circuit."""
    num_qubits = circuit.num_qubits
    num_entanglements = int(entangle_ratio * num_qubits)

    # Randomly select pairs to entangle from a cyclic list
    all_pairs = [(i, (i + 1) % num_qubits) for i in range(num_qubits)]
    selected_pairs = np.random.choice(len(all_pairs), num_entanglements, replace=False)

    # Add entanglement gates to circuit based on selected pairs
    entanglements_added = 0
    for index in selected_pairs:
        qubit1, qubit2 = all_pairs[index]
        circuit.cx(qubit1, qubit2)
        entanglements_added += 1

    return circuit, entanglements_added


def create_hea_with_entangle_ratio(num_qubits, depth, entangle_ratio):
    """Creates a hardware efficient ansatz using RX and RY gates for rotations with given depth."""
    circuit = QuantumCircuit(num_qubits)
    parameters = {
        i: [(Parameter(f"θx_{i}_{j}"), Parameter(f"θy_{i}_{j}"))
            for j in range(depth)]
        for i in range(num_qubits)
    }

    # Add parameterized rotation gates
    for d in range(depth):
        for qubit in range(num_qubits):
            theta_x, theta_y = parameters[qubit][d]
            circuit.rx(theta_x, qubit)
            circuit.ry(theta_y, qubit)

            num_entanglements = int(entangle_ratio * num_qubits)

            # Randomly select pairs to entangle based on the entangle_ratio
            all_pairs = [(i, i + 1) for i in range(num_qubits - 1)]
            # all_pairs.append((num_qubits-1, 0))
            print(all_pairs, num_entanglements)
            if num_entanglements > 0:
                selected_pairs = np.random.choice(len(all_pairs), num_entanglements, replace=False)

                # Add CNOT gates based on selected entangled pairs
                for index in selected_pairs:
                    qubit1, qubit2 = all_pairs[index]
                    circuit.cx(qubit1, qubit2)

        return circuit


def create_hea(num_qubits, depth, theta_list):
    circuit = QuantumCircuit(num_qubits)

    for d in range(depth):
        for i in range(num_qubits):
            circuit.add_gate(
                merge(RY(i, theta_list[2 * i + 2 * num_qubits * d]), RZ(i, theta_list[2 * i + 1 + 2 * num_qubits * d])))
        for i in range(num_qubits // 2):
            circuit.add_gate(CZ(2 * i, 2 * i + 1))
        for i in range(num_qubits // 2 - 1):
            circuit.add_gate(CZ(2 * i + 1, 2 * i + 2))
    for i in range(num_qubits):
        circuit.add_gate(
            merge(RY(i, theta_list[2 * i + 2 * num_qubits * depth]),
                  RZ(i, theta_list[2 * i + 1 + 2 * num_qubits * depth])))

    return circuit
