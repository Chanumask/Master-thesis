import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import COBYLA, SPSA, NELDER_MEAD

global_energy_history = []


def create_hea(num_qubits, depth, ansatz_id, ent_gate='cx'):
    circuit = QuantumCircuit(num_qubits)

    def add_entangling_gate(circ, control, target):
        if ent_gate == 'cx':
            circ.cx(control, target)
        elif ent_gate == 'cz':
            circ.cz(control, target)
        else:
            raise ValueError("ent_gate must be 'cx' or 'cz'.")

    if ansatz_id == 1:
        # Ansatz 1: typical HEA from many papers
        theta_list = [Parameter(f'θ_{i}_{j}') for i in range(num_qubits) for j in range(2 * depth)]

        for d in range(depth):
            for i in range(num_qubits):
                theta_ry = theta_list[2 * i + 2 * num_qubits * d]
                circuit.ry(theta_ry, i)
            for i in range(num_qubits):
                theta_rz = theta_list[2 * i + 1 + 2 * num_qubits * d]
                circuit.rz(theta_rz, i)

            for i in range(num_qubits - 1):
                add_entangling_gate(circuit, i, i + 1)
            add_entangling_gate(circuit, num_qubits - 1, 0)

    elif ansatz_id == 2:
        # Ansatz 2: circuit 2 as shown in the image from sim expressibility
        theta_list = [Parameter(f'θ_{i}_{j}') for i in range(num_qubits) for j in range(2 * depth)]

        for d in range(depth):
            for i in range(num_qubits):
                theta_rx = theta_list[2 * i + 2 * num_qubits * d]
                circuit.rx(theta_rx, i)
            for i in range(num_qubits):
                theta_rz = theta_list[2 * i + 1 + 2 * num_qubits * d]
                circuit.rz(theta_rz, i)

            for i in range(num_qubits - 1):
                add_entangling_gate(circuit, i, i + 1)
            add_entangling_gate(circuit, num_qubits - 1, 0)

    elif ansatz_id == 3:
        # Ansatz 3: Circuit 15 as shown in sim expressibility
        theta_list = [Parameter(f'θ_{i}_{j}') for i in range(num_qubits) for j in range(depth)]

        for d in range(depth):
            for i in range(num_qubits):
                theta_ry = theta_list[i + num_qubits * d]
                circuit.ry(theta_ry, i)
            if d % 2 == 0:
                # Even layers
                for i in range(0, num_qubits - 1, 2):
                    add_entangling_gate(circuit, i, i + 1)
            else:
                # Odd layers
                for i in range(1, num_qubits - 1, 2):
                    add_entangling_gate(circuit, i, i + 1)
                add_entangling_gate(circuit, num_qubits - 1, 0)
    else:
        raise ValueError("Invalid ansatz_id. Supported ids are 1 and 2.")

    random_params = np.random.uniform(-0.001, 0.001, len(theta_list))
    # circuit = circuit.assign_parameters({param: value for param, value in zip(theta_list, random_params)})
    return circuit, random_params


def create_test_ansatz(num_qubits, depth=2):
    circuit = QuantumCircuit(num_qubits)

    # Create parameters
    theta = [[Parameter(f"θ_{i}_{j}") for j in range(2)] for i in range(num_qubits)]

    # Add parametrized rotation gates and entanglement
    for d in range(depth):
        for i in range(num_qubits):
            circuit.ry(theta[i][0], i)
            circuit.rz(theta[i][1], i)
            circuit.x(i)

    # Add entangling CNOT gates between adjacent qubits
    for i in range(num_qubits - 1):
        circuit.cx(i, i + 1)
    print(circuit.draw())
    return circuit


def create_vqe(config, qubits, energy_history, num_iteration):
    ansatz_depth = config['vqe_parameters']['ansatz_layers']
    ansatz_type = config['vqe_parameters']['ansatz_type']

    entangle_method = config['vqe_parameters'].get('entangle_method', 'dropout')
    dropout_ratio = config['vqe_parameters']['dropout_ratio']
    param_theta = config['vqe_parameters'].get('param_theta', np.pi / 2)  # Default parameter for parametric entangling

    optimizer_choice = config['vqe_parameters'].get('optimizer', 'COBYLA')

    # Ansatz selection
    if ansatz_type.startswith('HEA'):
        try:
            ansatz_id = int(ansatz_type.split('_')[-1])
            ansatz, assigned_params = create_hea(qubits, ansatz_depth, ansatz_id, config['vqe_parameters'].get('ent_gate', 'cx'))
        except (ValueError, IndexError):
            raise ValueError(f"Invalid HEA ansatz type format: {ansatz_type}. Expected format: 'HEA_<int>'")
    else:
        raise ValueError(f"Unsupported ansatz type: {ansatz_type}")

    # Entanglement customization
    if entangle_method == 'dropout':
        ansatz = apply_dropout(ansatz, dropout_ratio)
    elif entangle_method == 'parametrize':
        ansatz = apply_parametric_entangling(ansatz, param_theta)  # Apply parameterized CZ with theta
    elif entangle_method == 'none':
        pass
    else:
        raise ValueError(f"Unsupported entangle method: {entangle_method}")

    estimator = Estimator()

    # Select the optimizer
    if optimizer_choice == 'COBYLA':
        optimizer = COBYLA()
    elif optimizer_choice == 'SPSA':
        optimizer = SPSA()
    elif optimizer_choice == 'NELDER_MEAD':
        optimizer = NELDER_MEAD()
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_choice}")

    def vqe_callback(eval_count, params, energy, metadata):
        if eval_count % 10 == 0:
            energy_history.append(float(round(energy, 4)))

    vqe = VQE(estimator, ansatz, optimizer, callback=vqe_callback)

    return ansatz, assigned_params, vqe


def apply_dropout(circuit, dropout_rate):
    """
    Applies dropout to the entangling gates (CZ gates) in a given quantum circuit.

    Args:
        circuit (QuantumCircuit): The input quantum circuit to which dropout will be applied.
        dropout_rate (float): The probability with which each CZ gate will be removed.

    Returns:
        QuantumCircuit: A modified quantum circuit with dropout applied to entangling gates.
    """
    modified_circuit = QuantumCircuit(circuit.num_qubits)

    for instruction, qubits, clbits in circuit.data:

        if instruction.name in ['cz', 'cx']:
            if np.random.rand() >= dropout_rate:
                modified_circuit.append(instruction, [q._index for q in qubits])
        else:
            modified_circuit.append(instruction, [q._index for q in qubits])

    return modified_circuit


def apply_parametric_entangling(circuit, theta_value):
    """
    Replaces CZ gates with parametrized Controlled-Z gates (and cx with crx respectively)
    to control entanglement strength.

    Args:
        circuit (QuantumCircuit): The input ansatz circuit to be modified.
        theta_value (float): The phase angle for the parametrized CZ gate (0 <= theta <= pi).

    Returns:
        QuantumCircuit: A modified quantum circuit with parametrized entangling gates.
    """
    modified_circuit = QuantumCircuit(circuit.num_qubits)
    theta_value = eval(theta_value)

    for instruction, qubits, clbits in circuit.data:
        if instruction.name == 'cz':
            modified_circuit.crz(theta_value, qubits[0]._index, qubits[1]._index)
        elif instruction.name == 'cx':
            modified_circuit.crx(theta_value, qubits[0]._index, qubits[1]._index)
        else:
            modified_circuit.append(instruction, [q._index for q in qubits])

    return modified_circuit


def add_cyclic_entanglement(circuit, dropout_ratio):
    """Add cyclic entanglement using CX gates at the beginning of the circuit."""
    num_qubits = circuit.num_qubits
    num_entanglements = int(dropout_ratio * num_qubits)

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
