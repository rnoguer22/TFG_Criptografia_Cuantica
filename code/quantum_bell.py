from qiskit.circuit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import SamplerV2
from qiskit_aer import AerSimulator

import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os



load_dotenv()

# Rotaciones para cada base
angles = {
    'a1': np.pi / 2,
    'a3': 0,
    'b1': np.pi / 4,
    'b3': -np.pi / 4
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
    qc.ry(- theta_a, 0)
    qc.ry(- theta_b, 1)
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
        ('a1', 'b1'),
        ('a1', 'b3'),
        ('a3', 'b1'),
        ('a3', 'b3')
    ]
    E = {}
    for a, b in settings:
        theta_a = angles[a]
        theta_b = angles[b]
        qc = create_chsh_circuit(theta_a, theta_b, eve)
        # qc.draw('mpl')
        # plt.show()
        # backend, circuit_isa = get_Backend_ISA(qc)
        # job = execute(backend, circuit_isa)
        '''job = get_Job(os.getenv('BELL_JOB_ID'))
        counts = job.result()[0].data.c.get_counts()'''
        result = simulator.run(qc, shots = 8192*2).result()
        counts = result.get_counts()
        E[(a, b)] = compute_expectation(counts)

    # Calcular S
    S = E[('a1', 'b1')] - E[('a1', 'b3')] + E[('a3', 'b1')] + E[('a3', 'b3')]
    print(counts, ' ', S, ' ', E, '\n')
    return S, E


def plot_correlations(correlations_dict: dict, eve: bool = True, path: str = ''):
    labels = list(correlations_dict.keys())
    values = [float(correlations_dict[k]) for k in labels]
    ideal = np.sqrt(2)/2

    plt.figure(figsize=(8, 5))
    bars = plt.bar([str(x) for x in labels], values, width=0.6, color='steelblue')
    # Añadimos los valores encima de las barras
    for bar in bars:
        height = bar.get_height()
        if height >= 0:
            height += 0.05
        else:
            height -= 0.05
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}',  # 2 decimales
                 ha='center', va='bottom' if height >=0 else 'top', fontsize=11)
    if not eve:
        plt.axhline(ideal, linestyle='dashed', color='red', label=r'Valor ideal $\frac{\sqrt{2}}{2}$')
        plt.axhline(-ideal, linestyle='dashed', color='red')

    plt.axhline(0, color='black', linewidth=0.5)
    plt.ylim(-1.1, 1.1)
    plt.title('Correlaciones $E(a_i, b_j)$')
    plt.ylabel('$E(a_i, b_j)$')
    plt.xlabel('Bases')
    plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), frameon=True, shadow=False, ncol=1)
    plt.grid(True, axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    if path:
        plt.savefig(path)
    # plt.show()


def plot_s_value(s_simulado, eve: bool = True, path: str = ''):
    s_ideal = 2 * np.sqrt(2)

    plt.figure(figsize=(4, 3))
    bars = plt.bar(['S simulado'], [s_simulado], width=0.2, color='skyblue')
    plt.xlim(-0.5, 0.5) 
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.05, f'{height:.2f}',  # 2 decimales
                    ha='center', va='bottom' if height >=0 else 'top', fontsize=11)
    if not eve:
        plt.axhline(s_ideal, linestyle='dashed', color='green', label='Valor ideal 2√2')
    plt.axhline(2, linestyle='dotted', color='red', label='Límite clásico (2)')
    plt.ylim(0, max(s_simulado, s_ideal) + 0.5)
    plt.title('Valor de S en la desigualdad de CHSH')
    plt.ylabel('S')
    plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), frameon=True, shadow=False, ncol=1)
    plt.tight_layout()
    if path:
        plt.savefig(path)
    # plt.show()


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
    plot_correlations(E_honest, eve=False, path='img/Simulation/corr_e91.png')
    plot_s_value(S_honest, eve=False, path='img/Simulation/s_e91.png')

    print("\n🚨 Caso con Eve:")
    print("Correlaciones:", E_eavesdropped)
    print("S =", S_eavesdropped)
    plot_correlations(E_eavesdropped, path='img/Simulation/corr_e91_eve.png')
    plot_s_value(S_eavesdropped, path='img/Simulation/s_e91_eve.png')