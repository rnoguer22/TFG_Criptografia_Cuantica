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
    qc.h(0)
    qc.cx(0, 1)
    if eve:
        # Eve intercepta y mide el qubit de Alice, destruyendo el entrelazamiento
        qc.measure(0, 0) 
        qc.reset(0)      # Eve resetea el qubit 0 a |0>
        qc.barrier()
    # Aplicamos las rotaciones de las bases que Alice y Bob han elegido
    qc.ry(- theta_a, 0)
    qc.ry(- theta_b, 1)
    qc.measure(0, 0)
    qc.measure(1, 1)

    return qc
    

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
def compute_S(eve: bool = False):
    # Usamos un simulador sin ruido
    simulator = AerSimulator()
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
        # Para cada angulo de las bases de Alice y Bob, creamos un circuito y lo ejecutamos para calcular la correlacion
        qc = create_chsh_circuit(theta_a, theta_b, eve)
        '''if count == 0:
            qc.draw('mpl')
            plt.savefig(circuit_path)'''
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
    print(all_counts_data, ' ', S, ' ', E, '\n')
    return all_counts_data, S, E


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
def plot_measurement_histogram(all_counts_data: dict, path: str = ''):
    possible_outcomes = ['00', '01', '10', '11']    
    base_pairs = list(all_counts_data.keys())
    
    reorganized_data = {outcome: [] for outcome in possible_outcomes}

    for base_pair in base_pairs:
        counts = all_counts_data[base_pair]
        for outcome in possible_outcomes:
            reorganized_data[outcome].append(counts.get(outcome, 0)) # Obtener el conteo o 0 si no existe

    # Configuramos las barras del grafico
    num_base_pairs = len(base_pairs)
    bar_width = 0.2  # Ancho de cada sub-barra
    
    # Posiciones para las distintas agrupaciones de barras en el gráfico
    indices = np.arange(len(possible_outcomes))
    plt.figure(figsize=(12, 7))
    # Colores
    colors = plt.cm.viridis(np.linspace(0, 1, num_base_pairs))

    # Crearmos cada barra del par de bases de cada resultado de la medición
    for i, base_pair in enumerate(base_pairs):
        x_positions = indices + (i - num_base_pairs / 2 + 0.5) * bar_width
        # Frecuencias para cada par de bases
        frequencies_for_this_pair = [all_counts_data[base_pair].get(outcome, 0) for outcome in possible_outcomes]  
        # Dibujamos las barras
        plt.bar(x_positions, frequencies_for_this_pair, width=bar_width, 
                label=base_pair, color=colors[i])

    plt.title('Histograma de los Resultados de las Mediciones')
    plt.xlabel('Resultado de las Mediciones de Bob y Alice')
    plt.ylabel('Frecuencia')
    plt.xticks(indices, possible_outcomes) 
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(title='Par de Bases')
    plt.tight_layout()

    if path:
        plt.savefig(path)
    # plt.show()




if __name__ == '__main__':

    all_counts_honest, S_honest, E_honest = compute_S(eve=False)
    all_counts_eavesdropped, S_eavesdropped, E_eavesdropped = compute_S(eve=True)

    print("📡 Caso sin Eve:")
    print("Correlaciones:", E_honest)
    print("S =", S_honest)
    plot_correlations(E_honest, eve=False, path='img/Simulation/e91/corr_e91.png')
    plot_s_value(S_honest, eve=False, path='img/Simulation/e91/s_e91.png')
    plot_measurement_histogram(all_counts_data=all_counts_honest, path='img/Simulation/e91/histograma_medidas.png')

    print("\n🚨 Caso con Eve:")
    print("Correlaciones:", E_eavesdropped)
    print("S =", S_eavesdropped)
    plot_correlations(E_eavesdropped, path='img/Simulation/e91/corr_e91_eve.png')
    plot_s_value(S_eavesdropped, path='img/Simulation/e91/s_e91_eve.png')
    plot_measurement_histogram(all_counts_data=all_counts_eavesdropped, path='img/Simulation/e91/histograma_medidas_eve.png')
