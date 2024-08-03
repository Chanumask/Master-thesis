from qiskit import transpile
from qiskit_aer import AerSimulator


def run_circuit(circuit):
    simulator = AerSimulator()
    result = simulator.run(transpile(circuit, simulator), shots=1024).result()
    counts = result.get_counts(circuit)
    return counts
