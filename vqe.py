import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import CRXGate
from qiskit.primitives import Estimator
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA, SPSA, NELDER_MEAD

from eval import calculate_cut_cost, get_probabilities, calculate_ansatz_metrics
from max_cut import get_hamiltonian, brute_force_max_cut
from util import Result


def create_hea(num_qubits, depth, ansatz_id):
    """
    Creates the ansatz specified by id and returns the ansatz and a set of random parameter but doesn't assign them yet.
    """
    circuit = QuantumCircuit(num_qubits)

    if ansatz_id == 1:
        # Ansatz 1: single rotational gate RY ansatz (eg. Rohe et al)
        theta_list = [Parameter(f'θ_{i}_{j}') for i in range(num_qubits) for j in range(depth)]

        for d in range(depth):
            for i in range(num_qubits):
                circuit.ry(theta_list[i + num_qubits * d], i)

            for i in range(num_qubits - 1):
                circuit.cx(i, i + 1)
            circuit.cx(num_qubits - 1, 0)

            circuit.barrier()


    elif ansatz_id == 2:
        # Ansatz 2: double rotational ansatz RYRX ansatz from Sim et. al and found in many other papers (high expr.)
        theta_list = [Parameter(f'θ_{i}_{j}') for i in range(num_qubits) for j in range(2 * depth)]

        for d in range(depth):
            for i in range(num_qubits):
                circuit.ry(theta_list[2 * i + 2 * num_qubits * d], i)
                circuit.rx(theta_list[2 * i + 1 + 2 * num_qubits * d], i)

            for i in range(num_qubits - 1):
                circuit.cx(i, i + 1)
            circuit.cx(num_qubits - 1, 0)

            circuit.barrier()

    elif ansatz_id == 3:
        # Ansatz 3: Uneven entanglement structure, Circuit 15 as shown in sim expressibility (low expr.)
        theta_list = [[Parameter(f'θ_{i}{j}') for j in range(2)] for i in range(num_qubits)]

        for d in range(depth):
            for i in range(num_qubits):
                circuit.ry(theta_list[i][0], i)

            for i in range(0, num_qubits - 1, 2):
                circuit.cx(i, i + 1)

            circuit.barrier()

            for i in range(num_qubits):
                circuit.ry(theta_list[i][1], i)

            for i in range(1, num_qubits - 1, 2):
                circuit.cx(i, i + 1)
            circuit.cx(num_qubits - 1, 0)

            circuit.barrier()

    else:
        raise ValueError("Invalid ansatz_id. Supported ids are 1, 2 and 3.")

    random_params = np.random.uniform(-0.001, 0.001, len(theta_list))
    return circuit, random_params


def apply_dropout(circuit, dropout_rate):
    """
    Applies dropout to the entangling gates (CZ gates) in a given quantum circuit.

    Args:
        circuit (QuantumCircuit): The input quantum circuit to which dropout will be applied.
        dropout_rate (float): The probability with which each CZ gate will be removed.

    Returns:
        QuantumCircuit: A modified quantum circuit with dropout applied to entangling gates.
    """
    # initial_cx_count = sum(1 for inst, _, _ in circuit.data if inst.name == 'cx')
    # print(circuit.draw())

    keep_gate = np.random.rand(len(circuit.data)) >= dropout_rate

    circuit.data = [
        (inst, qargs, cargs) for (inst, qargs, cargs), keep in zip(circuit.data, keep_gate)
        if keep or inst.name not in ['cx']
    ]
    # final_cx_count = sum(1 for inst, _, _ in circuit.data if inst.name == 'cx')
    # print(f"\nDropout Rate: {dropout_rate}")
    # print(f"CX Gates Removed: {initial_cx_count - final_cx_count} / {initial_cx_count}")
    # print("\nModified Circuit:")
    # print(circuit.draw())
    return circuit


def apply_constant_parametric_entangling(circuit, theta_value):
    """
    Replaces CX gates with parametrized CRX gates to control entanglement strength.

    Args:
        circuit (QuantumCircuit): The input ansatz circuit (modified in-place).
        theta_value (str or float): The rotation angle for the parametrized CRX gate.

    Returns:
        QuantumCircuit: The modified quantum circuit.
    """
    # Convert theta_value safely
    if isinstance(theta_value, str):
        try:
            theta_value = eval(theta_value, {"np": np})  # Safe eval with restricted namespace
        except Exception:
            raise ValueError(f"Invalid theta_value: {theta_value}. Must be a valid numerical expression.")

    theta_value = float(theta_value)

    # print("\nOriginal Circuit:")
    # print(circuit.draw())

    new_data = []
    lambda_param = Parameter('λ')  # Free parameter for symmetric mapping
    for inst, qargs, cargs in circuit.data:
        if inst.name == 'cx':
            symmetric_theta = theta_value * (2 / (1 + (lambda_param ** 2)) - 1)
            crx_gate = CRXGate(symmetric_theta)
            new_data.append((crx_gate, qargs, cargs))
        else:
            new_data.append((inst, qargs, cargs))

    # Replace circuit data with modified version
    circuit.data = new_data

    # print(f"\nTheta Value: {theta_value}")
    # print("\nModified Circuit:")
    # print(circuit.draw())

    return circuit


def apply_parametric_entangling(circuit, theta_value):
    """
    Replaces CX gates with parametrized CRX gates to control entanglement strength
    while ensuring parameter ordering and correct bounds also for multiple layers.
    """
    # Convert theta_value safely
    if isinstance(theta_value, str):
        try:
            theta_value = eval(theta_value, {"np": np})  # Safe eval with restricted namespace
        except Exception:
            raise ValueError(f"Invalid theta_value: {theta_value}. Must be a valid numerical expression.")

    theta_max = float(theta_value)

    crx_param_bounds = {}
    parameter_bounds = []  # Ordered list of parameter bounds to pass to vqe

    new_data = []
    crx_param_index = 0

    # print("\nOriginal Circuit:")
    # print(circuit.draw())

    for inst, qargs, cargs in circuit.data:
        if inst.name in ['rx', 'ry', 'rz']:
            # RX, RY, RZ gates: Keep them unbounded
            param = Parameter(f'θ_{inst.name}_{len(parameter_bounds)}')
            parameter_bounds.append(None)
            new_data.append((inst.__class__(param), qargs, cargs))

        elif inst.name == 'cx':
            # Replace CX with a trainable CRX gate and create corresponding bounds
            crx_param = Parameter(f'θ_crx_{crx_param_index}')
            crx_gate = CRXGate(crx_param)

            crx_param_bounds[crx_param] = (-theta_max, theta_max)
            parameter_bounds.append((-theta_max, theta_max))

            new_data.append((crx_gate, qargs, cargs))
            crx_param_index += 1
        else:
            new_data.append((inst, qargs, cargs))

    circuit.data = new_data
    return circuit, crx_param_bounds, parameter_bounds


def create_vqe(config, qubits, energy_history, num_iteration):
    ansatz_depth = config['vqe_parameters']['ansatz_layers']
    ansatz_type = config['vqe_parameters']['ansatz_type']

    entangle_method = config['vqe_parameters']['entangle_method']
    dropout_ratio = config['vqe_parameters']['dropout_ratio']
    param_theta = config['vqe_parameters']['param_theta']

    optimizer_choice = config['vqe_parameters']['optimizer']

    # Ansatz selection
    if ansatz_type.startswith('HEA'):
        ansatz_id = int(ansatz_type.split('_')[-1])
        ansatz, random_params = create_hea(qubits, ansatz_depth, ansatz_id)
    else:
        raise ValueError(f"Unsupported ansatz type: {ansatz_type}")

    # Entanglement customization
    if entangle_method == 'dropout':
        ansatz = apply_dropout(ansatz, dropout_ratio)
    elif entangle_method == 'parametrize':
        ansatz, crx_param_bounds, parameter_bounds = apply_parametric_entangling(ansatz, param_theta)
        if parameter_bounds is None:
            raise ValueError("Error: parameter_bounds should not be None after applying parametric entangling.")
        ansatz.parameter_bounds = [
            bound if bound is not None else (-2 * np.pi, 2 * np.pi) for bound in parameter_bounds
        ]
        # print("Parameter Bounds Set on Ansatz:", ansatz.parameter_bounds)
    elif entangle_method == 'none':
        pass
    else:
        raise ValueError(f"Unsupported entangle method: {entangle_method}")

    # Select the optimizer
    optimizer_mapping = {'COBYLA': COBYLA(), 'SPSA': SPSA(), 'NELDER_MEAD': NELDER_MEAD()}
    optimizer = optimizer_mapping.get(optimizer_choice)
    if optimizer is None:
        raise ValueError(f"Unsupported optimizer: {optimizer_choice}")

    estimator = Estimator()


    def vqe_callback(eval_count, params, energy, metadata):
        if eval_count % 10 == 0:
            energy_history.append(float(round(energy, 4)))

    vqe = VQE(estimator, ansatz, optimizer, callback=vqe_callback)

    return ansatz, random_params, vqe

def run_vqe_for_max_cut(g, config, seed, num_iteration):
    """Solve the MaxCut problem on a given NetworkX graph using VQE."""
    start_time = time.time()

    qubits = g.number_of_nodes()
    hamiltonian, offset = get_hamiltonian(g)
    energy_history = []

    ansatz, random_params, vqe = create_vqe(config, qubits, energy_history, num_iteration)

    # calculate expressibility and entangling capability only in the first iteration
    entanglement_metrics = {}
    if num_iteration == 0:
        entanglement_metrics = calculate_ansatz_metrics(ansatz, config)

    # Run optimization
    vqe_result = vqe.compute_minimum_eigenvalue(hamiltonian)
    optimized_ansatz = ansatz.assign_parameters(vqe_result.optimal_parameters)

    probabilities = get_probabilities(optimized_ansatz)

    print(entanglement_metrics)

    solution = max(probabilities.items(), key=lambda x: x[1])[0]
    cut_cost = calculate_cut_cost(g, solution)
    optimal_cut_value = brute_force_max_cut(g)

    iteration_time = time.time() - start_time

    result = Result(
        graph_edges=list(g.edges()),
        optimal_value=vqe_result.optimal_value,
        distribution=probabilities,
        solution=solution,
        cut_cost=cut_cost,
        brute_forced_cost=optimal_cut_value,
        entanglement_metrics=entanglement_metrics,
        energy_history=energy_history,
        seed=seed,
        iter_time=iteration_time
    )
    result.compute_cut_cost_difference()
    result.compute_approximation_ratio()

    print(f"approx. ratio: {result.approximation_ratio}")
    return result