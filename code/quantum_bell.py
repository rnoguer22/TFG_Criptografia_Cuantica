from qiskit.circuit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import SamplerV2
from qiskit_aer import Aer, AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os



load_dotenv()

# Rotaciones para cada base
angles = {
    'A0': 0,
    'A1': np.pi / 4,
    'B0': np.pi / 8,
    'B1': -np.pi / 8
}

# Crea el circuito CHSH
def create_chsh_circuit(theta_a, theta_b, eve: bool = False):
    qc = QuantumCircuit(2, 2)

    # Crear estado de Bell |Φ+>
    qc.h(0)
    qc.cx(0, 1)
    if eve:
        # Eve intercepta y mide Q0 (de Alice), destruyendo el entrelazamiento
        qc.measure(0, 0)
        qc.reset(0)
        # Después envía un nuevo qubit |0⟩ sin entrelazar
        qc.barrier()
    # Aplicar rotaciones de base
    qc.ry(-2 * theta_a, 0)
    qc.ry(-2 * theta_b, 1)
    # Medición final
    qc.measure(0, 0)
    qc.measure(1, 1)

    return qc


# Obtenemos el backend y el ISA
def get_Backend_ISA(circuit):
    service = QiskitRuntimeService(channel="ibm_quantum")
    # El backend sera el procesador menos ocupado en el momento
    backend = service.least_busy(min_num_qubits=100, operational=True)
    # pass_manager
    pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
    # circuito ISA (Instruction Set Architecture)
    circuit_isa = pm.run(circuit)
    return backend, circuit_isa
    

# Ejecutamos el circuito con un IBM Quantum
def execute(backend, circuit_isa, shots: int = 8192):
    sampler = SamplerV2(backend)
    sampler.options.default_shots = shots
    sampler.options.dynamical_decoupling.enable = True          # Supresión de errores: Dynamical Decoupling
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.twirling.enable_gates = True                # Supresión de errores: Pauli Twirling
    pub = (circuit_isa)
    job = sampler.run([pub])        
    return job


# Calcula E(a,b)
def compute_expectation(counts):
    shots = sum(counts.values())
    expectation = 0
    for outcome, count in counts.items():
        bit_a = int(outcome[-1])  # Qubit 0 es el último bit en Qiskit
        bit_b = int(outcome[-2])  # Qubit 1 es el penúltimo bit
        parity = (-1) ** (bit_a + bit_b)  # Cambia XOR por suma para correlación CHSH
        expectation += parity * count / shots
    return expectation


# Ejecuta todos los circuitos y calcula S
def compute_S(eve=False):
    # backend = Aer.get_backend('qasm_simulator')
    simulator = AerSimulator()  # ← Simulador sin ruido
    settings = [
        ('A0', 'B0'),
        ('A0', 'B1'),
        ('A1', 'B0'),
        ('A1', 'B1')
    ]
    E = {}
    for a, b in settings:
        print('\n', a, ' ', b, '\n')
        theta_a = angles[a]
        theta_b = angles[b]
        qc = create_chsh_circuit(theta_a, theta_b, eve)
        # qc.draw('mpl')
        # plt.show()
        # backend, circuit_isa = get_Backend_ISA(qc)
        # job = execute(backend, circuit_isa)
        '''job = get_Job(os.getenv('BELL_JOB_ID'))
        counts = job.result()[0].data.c.get_counts()'''
        result = simulator.run(qc, shots=8192).result()
        counts = result.get_counts()
        E[(a, b)] = compute_expectation(counts)

    print(counts)
    # Calcular S
    S = E[('A0', 'B0')] + E[('A0', 'B1')] + E[('A1', 'B0')] - E[('A1', 'B1')]
    print(S)
    return S, E


def get_Job(job_id):
    service = QiskitRuntimeService(channel="ibm_quantum")
    job = service.job(job_id) 
    return job




if __name__ == '__main__':

    S_honest, E_honest = compute_S(eve=False)
    S_eavesdropped, E_eavesdropped = compute_S(eve=True)

    print("📡 Caso sin Eve:")
    print("Correlaciones:", E_honest)
    print("S =", S_honest)

    print("\n🚨 Caso con Eve:")
    print("Correlaciones:", E_eavesdropped)
    print("S =", S_eavesdropped)