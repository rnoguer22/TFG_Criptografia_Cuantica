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

# Angulos de las bases en el plano XZ que Alice y Bob van a usar para obtener el valor maximo de S
angles = {
    'a1': np.pi / 2,
    'a3': 0,
    'b1': np.pi / 4,
    'b3': -np.pi / 4
}

# Funcion para crear el circuito
def create_chsh_circuit(theta_a, theta_b, eve: bool = False):
    qc = QuantumCircuit(2, 2)
    # Creamos el estado de Bell |Φ+>
    qc.h(0)
    qc.cx(0, 1)
    if eve:
        # Eve intercepta y mide Q0 (de Alice), destruyendo el entrelazamiento
        qc.measure(0, 0)
        # Después envía un nuevo qubit |0⟩ sin entrelazar
        qc.barrier()
    # Aplicamos las rotaciones de las bases que Alice y Bob han elegido
    qc.ry(- theta_a, 0)
    qc.ry(- theta_b, 1)
    qc.measure(0, 0)
    qc.measure(1, 1)

    return qc


# Obtenemos el backend y el ISA
def get_Backend_ISA(circuit):
    service = QiskitRuntimeService(channel="ibm_quantum")
    # El backend sera el procesador menos ocupado en el momento
    backend = service.least_busy(min_num_qubits=100, operational=True)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
    # circuito ISA (Instruction Set Architecture)
    circuit_isa = pm.run(circuit)
    return backend, circuit_isa
    

# Ejecutamos el circuito con IBM Quantum
def execute(backend, circuit_isa, shots: int = 8192):
    sampler = SamplerV2(backend)
    sampler.options.default_shots = shots
    sampler.options.dynamical_decoupling.enable = True          
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.twirling.enable_gates = True               
    pub = (circuit_isa)
    job = sampler.run([pub])        
    return job


# Con esta funcion calculamos E(a,b)
def compute_expectation(counts):
    shots = sum(counts.values())
    expectation = 0
    for outcome, count in counts.items():
        bit_a = int(outcome[-1])  
        bit_b = int(outcome[-2])  
        parity = (-1) ** (bit_a + bit_b)  
        expectation += parity * count / shots
    return expectation


# Ejecutamos todo y calculamos S
def compute_S(circuit_path: str, eve: bool = False):
    # backend = Aer.get_backend('qasm_simulator')
    simulator = AerSimulator()  # ← Simulador sin ruido
    settings = [
        ('a1', 'b1'),
        ('a1', 'b3'),
        ('a3', 'b1'),
        ('a3', 'b3')
    ]

    E = {}
    all_counts_data = {} 

    count = 0
    for a, b in settings:
        theta_a = angles[a]
        theta_b = angles[b]
        qc = create_chsh_circuit(theta_a, theta_b, eve)
        if count == 0:
            qc.draw('mpl')
            plt.savefig(circuit_path)
        # backend, circuit_isa = get_Backend_ISA(qc)
        # job = execute(backend, circuit_isa)
        '''job = get_Job(os.getenv('BELL_JOB_ID'))
        counts = job.result()[0].data.c.get_counts()'''
        result = simulator.run(qc, shots = 8192).result()
        counts = result.get_counts()
        key_str = f"{a}{b}"
        all_counts_data[key_str] = counts
        E[(a, b)] = compute_expectation(counts)

    # Calculamos S
    S = E[('a1', 'b1')] - E[('a1', 'b3')] + E[('a3', 'b1')] + E[('a3', 'b3')]
    print(counts, ' ', S, ' ', E, '\n')
    return all_counts_data, counts, S, E


# Mostramos un grafico de barras con las correlaciones entre las bases de Alice y Bob
def plot_correlations(correlations_dict: dict, eve: bool = True, path: str = ''):
    labels = list(correlations_dict.keys())
    values = [float(correlations_dict[k]) for k in labels]
    ideal = np.sqrt(2)/2

    plt.figure(figsize=(8, 5))
    bars = plt.bar([str(x) for x in labels], values, width=0.6, color='steelblue')
    # Añadimos los valores encima de las barras
    for bar in bars:
        height = bar.get_height()
        original_height = height
        if height >= 0:
            height += 0.05
        else:
            height -= 0.05
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{original_height:.2f}',  # Tomamos solamente 2 decimales
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


# Grafico para ver mejor el valor de S
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
        plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), frameon=True, shadow=False, ncol=1)
    else:
        plt.axhline(2, linestyle='dotted', color='red', label='Límite clásico (2)')
        plt.legend()
    plt.ylim(0, max(s_simulado, s_ideal) + 0.5)
    plt.title('Valor de S en la desigualdad de CHSH')
    plt.ylabel('S')
    plt.tight_layout()
    if path:
        plt.savefig(path)
    # plt.show()


# Mostramos en otro grafico los qubits de Alice y Bob una vez ya han colapsado y finalizado el protocolo
def plot_measurement_histogram(counts, path: str = ''):
    keys = sorted(counts.keys()) 
    values = [counts[key] for key in keys]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(keys, values, color='mediumslateblue')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.05, height,
                    ha='center', va='bottom' if height >=0 else 'top', fontsize=11)

    plt.title('Histograma de resultados de medición')
    plt.xlabel('Resultado (Alice y Bob)')
    plt.ylabel('Frecuencia')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    if path:
        plt.savefig(path)
    # plt.show()


# Funcion para obtener el job_id de una simulacion de IBM Quantum
def get_Job(job_id):
    service = QiskitRuntimeService(channel="ibm_quantum")
    job = service.job(job_id) 
    return job




if __name__ == '__main__':

    all_counts_honest, counts_honest, S_honest, E_honest = compute_S(circuit_path='img/Simulation/e91/circuit.png', eve=False)
    all_counts_eavesdropped, counts_eavesdropped, S_eavesdropped, E_eavesdropped = compute_S(circuit_path='img/Simulation/e91/circuit_eve.png', eve=True)

    print("📡 Caso sin Eve:")
    print("Correlaciones:", E_honest)
    print("S =", S_honest)
    plot_correlations(E_honest, eve=False, path='img/Simulation/e91/corr_e91.png')
    plot_s_value(S_honest, eve=False, path='img/Simulation/e91/s_e91.png')
    plot_measurement_histogram(counts=counts_honest, path='img/Simulation/e91/histograma_medidas.png')

    print("\n🚨 Caso con Eve:")
    print("Correlaciones:", E_eavesdropped)
    print("S =", S_eavesdropped)
    plot_correlations(E_eavesdropped, path='img/Simulation/e91/corr_e91_eve.png')
    plot_s_value(S_eavesdropped, path='img/Simulation/e91/s_e91_eve.png')
    plot_measurement_histogram(counts=counts_eavesdropped, path='img/Simulation/e91/histograma_medidas_eve.png')
