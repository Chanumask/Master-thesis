import random

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


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


def create_hea(num_qubits, depth):
    circuit = QuantumCircuit(num_qubits)

    # Generate a list of 2 * num_qubits * (depth + 1) random integers between 0 and 100
    theta_list = [random.randint(0, 100) for _ in range(2 * num_qubits * (depth + 1))]

    for d in range(depth):
        for i in range(num_qubits):
            theta_ry = (theta_list[2 * i + 2 * num_qubits * d] / 100) * 2 * np.pi
            theta_rz = (theta_list[2 * i + 1 + 2 * num_qubits * d] / 100) * 2 * np.pi
            circuit.ry(theta_ry, i)
            circuit.rz(theta_rz, i)

        for i in range(num_qubits // 2):
            circuit.cz(2 * i, 2 * i + 1)
        for i in range(num_qubits // 2 - 1):
            circuit.cz(2 * i + 1, 2 * i + 2)

    for i in range(num_qubits):
        theta_ry = (theta_list[2 * i + 2 * num_qubits * depth] / 100) * 2 * np.pi
        theta_rz = (theta_list[2 * i + 1 + 2 * num_qubits * depth] / 100) * 2 * np.pi
        circuit.ry(theta_ry, i)
        circuit.rz(theta_rz, i)

    return circuit
