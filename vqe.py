import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import COBYLA, SPSA, NELDER_MEAD

from util import generate_qpic_circuit

global_energy_history = []


def create_hea(num_qubits, depth):
    circuit = QuantumCircuit(num_qubits)

    # Create parameter objects for each rotation angle
    theta_list = [Parameter(f'θ_{i}_{j}') for i in range(num_qubits) for j in range(2 * (depth + 1))]

    # Loop over depth and apply RY, RZ gates with parameterized angle (2 params per qubit per layer)
    for d in range(depth):
        for i in range(num_qubits):
            theta_ry = theta_list[2 * i + 2 * num_qubits * d]
            theta_rz = theta_list[2 * i + 1 + 2 * num_qubits * d]
            circuit.ry(theta_ry, i)
            circuit.rz(theta_rz, i)

        # Apply circular CZ entanglement
        for i in range(num_qubits - 1):
            circuit.cz(i, i + 1)
        circuit.cz(num_qubits - 1, 0)

    # Apply final RY, RZ gates with parameterized angles
    for i in range(num_qubits):
        theta_ry = theta_list[2 * i + 2 * num_qubits * depth]
        theta_rz = theta_list[2 * i + 1 + 2 * num_qubits * depth]
        circuit.ry(theta_ry, i)
        circuit.rz(theta_rz, i)

    return circuit


def create_simple_ansatz(num_qubits, depth=2):
    print("simple")
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


def create_vqe(config, qubits):
    ansatz_depth = config['vqe_parameters']['ansatz_layers']
    ansatz_type = config['vqe_parameters'].get('ansatz_type', 'RealAmplitudes')
    plot_ansatz = config['vqe_parameters']['plot_ansatz']

    entangle_method = config['vqe_parameters'].get('entangle_method', 'dropout')
    entangle_ratio = config['vqe_parameters']['entangle_ratio']
    param_theta = config['vqe_parameters'].get('param_theta', np.pi / 2)  # Default parameter for parametric entangling

    optimizer_choice = config['vqe_parameters'].get('optimizer', 'COBYLA')

    # Ansatz selection
    if ansatz_type == 'RealAmplitudes':
        ansatz = RealAmplitudes(num_qubits=qubits, reps=ansatz_depth)
    elif ansatz_type == 'EfficientSU2':
        ansatz = EfficientSU2(num_qubits=qubits, reps=ansatz_depth)
    elif ansatz_type == 'HEA':
        ansatz = create_hea(qubits, ansatz_depth)
    elif ansatz_type == 'Test':
        ansatz = create_simple_ansatz(qubits)
    else:
        raise ValueError(f"Unsupported ansatz type: {ansatz_type}")

    # Add Entanglement or introduce dropout
    full_circuit = QuantumCircuit(qubits)
    # full_circuit, n_entanglements = add_cyclic_entanglement(full_circuit, entangle_ratio)
    full_circuit.compose(ansatz, inplace=True)
    # print(full_circuit.decompose().draw())

    # Entanglement customization
    if entangle_method == 'dropout':
        ansatz = apply_dropout(ansatz, entangle_ratio)
    elif entangle_method == 'parametrize':
        ansatz = apply_parametric_entangling(ansatz, param_theta)  # Apply parameterized CZ with theta
    else:
        raise ValueError(f"Unsupported entangle method: {entangle_method}")

    if plot_ansatz:
        qpic_str = generate_qpic_circuit(ansatz)
        with open('circuit.qpic', 'w', encoding='utf-8') as file:
            file.write(qpic_str)
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

    vqe = VQE(estimator, full_circuit, optimizer)

    return ansatz, vqe


def vqe_callback(eval_count, params, energy, metadata):
    """Callback function to store the energy at each iteration."""
    global global_energy_history
    if eval_count % 10 == 0:  # only save every 10th value
        global_energy_history.append(float(round(energy, 4)))


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
        elif instruction.name =='cx':
            modified_circuit.crx(theta_value, qubits[0]._index, qubits[1]._index)
        else:
            modified_circuit.append(instruction, [q._index for q in qubits])

    return modified_circuit


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
