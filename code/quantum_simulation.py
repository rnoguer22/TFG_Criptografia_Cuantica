from qiskit.circuit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit_ibm_runtime import SamplerV2
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer import AerSimulator

import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np



class Quantum_Simulation:

    def __init__(self):
        load_dotenv()
    

    # Metodo para definir el circuito
    def define_Circuit(self):
        qreg_q = QuantumRegister(2, 'q')
        creg_c = ClassicalRegister(2, 'c')
        circuit = QuantumCircuit(qreg_q, creg_c)

        circuit.h(qreg_q[0])
        circuit.cx(qreg_q[0], qreg_q[1])
        circuit.measure(qreg_q[0], creg_c[0])
        circuit.measure(qreg_q[1], creg_c[1])

        return circuit
    

    # Metodo para dibujar y guardar el circuito
    def draw_Circuit(self, circuit, path: str = None):
        circuit.draw('mpl')
        if path is not None:
            plt.savefig(path)
        plt.show()

    
    # Obtenemos el backend y el ISA
    def get_Backend_ISA(self, circuit):
        service=QiskitRuntimeService(channel="ibm_quantum")
        # El backend sera el procesador menos ocupado en el momento
        backend = service.least_busy(min_num_qubits=100, operational=True)
        # pass_manager
        pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
        # circuito ISA (Instruction Set Architecture)
        circuit_isa = pm.run(circuit)

        return backend, circuit_isa


    def get_Characteristics(self, circuit_isa):
            print("\nCaracterísticas del circuito cuántico")
            print("  Depth:", circuit_isa.depth())
            print("  Cantidad de qubits:", circuit_isa.num_qubits)
            print("  Operaciones:", dict(circuit_isa.count_ops()))
            print("  Cantidad de operaciones multi-qubit: ", circuit_isa.num_nonlocal_gates(), "\n")
    

    # Ejecutamos el circuito con un IBM Quantum
    def execute(self, backend, circuit_isa):
        sampler = SamplerV2(backend)
        sampler.options.default_shots = 4096
        sampler.options.dynamical_decoupling.enable = True          # Supresión de errores: Dynamical Decoupling
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
        sampler.options.twirling.enable_gates = True                # Supresión de errores: Pauli Twirling
        pub = (circuit_isa)
        job = sampler.run([pub])
        
        return job
    

    # Metodo para seguir con un job que se ha quedado pendiente
    def continue_Job(self, job_id):
        service = QiskitRuntimeService(channel="ibm_quantum")
        job = service.job(job_id)
        return job
    

    # Metodo para mostrar los resultados
    def plot_Results(self, job, probs: bool = False, path: str = None):
        # Imprimimos los resultados obtenidos
        resultado = job.result()
        distribucion = resultado[0].data.c.get_counts()
        print('\n', distribucion, '\n')
        maximo = max(distribucion.items(), key=lambda x: x[1])
        print("Maximo :", maximo)

        if probs:
            # Ploteamos la distribución de probabilidades obtenida
            shots = sum(distribucion.values())
            distribucion_probas = {key: val/shots for key, val in distribucion.items()}
            plot_histogram(distribucion_probas, title = 'Distribución de Probabilidades') 
        else:
            plot_histogram(distribucion, title = 'Distribución de los Resultados')
        
        if path:
            plt.savefig(path)
        plt.show()

    
    # Metodo para mostrar el nivel de depolarizacion de error 
    def plot_Noise_Model(self, circuit: QuantumCircuit, alphas: list = [0, 0.25, 0.50], shots: int = 4096, path: str = None):
        counts_list = []
        legends = []

        # Esquema de colores
        cmap = cm.get_cmap('viridis')
        # Creamos un color distinto para cada alpha
        colors = [cmap(i) for i in np.linspace(0, 1, len(alphas))]

        for alpha in alphas:
            print(f"\nSimulación alpha = {alpha}")

            noise_model = NoiseModel()
            
            single_qubit_error = depolarizing_error(alpha, 1)
            noise_model.add_all_qubit_quantum_error(single_qubit_error, ['h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg', 'u', 'u1', 'u2', 'u3', 'rx', 'ry', 'rz'])
            
            two_qubit_error = depolarizing_error(alpha, 2) 
            noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx', 'cz', 'iswap', 'rxx', 'ryy', 'rzz', 'ecr'])
            
            simulator = AerSimulator(noise_model=noise_model)
            
            try:
                # Transpilamos el circuito original de 2 qubits para el AerSimulator
                transpiled_circuit = transpile(circuit, backend=simulator, optimization_level=0)

                result = simulator.run(transpiled_circuit, shots=shots).result()
                counts = result.get_counts(0) 
                counts_list.append(counts)
                legends.append(f'α={alpha}')
                print(f"Simulation for alpha={alpha} completed. Counts: {counts}")

            except Exception as e:
                print(f"Error en la simulación de alpha = {alpha}: {e}")
                counts_list.append({}) 
                legends.append(f'α={alpha} (Error)')
                continue 

        if counts_list: 
            fig = plot_histogram(counts_list, legend=legends, title="Frecuencias de Medición vs. Nivel de Ruido", color=colors)
            if path is not None:
                plt.savefig(path)
            plt.show()
            plt.close(fig)
        else:
            print("Error en la simulación")

    
    def plot_QBER_vs_Alpha(self, circuit: QuantumCircuit, alphas: list = np.linspace(0, 1, 11), shots: int = 1000, path: str = None):
        # Los valores ideales, sin ruido, son un 50, 50 para cada estado en nuestro caso
        ideal_counts = {'00': shots/2, '11': shots/2}  # Para un circuito Bell perfecto
        qber_values = []
        
        for alpha in alphas:
            # Configuramos el modelo de ruido
            noise_model = NoiseModel()
            single_qubit_error = depolarizing_error(alpha, 1)
            two_qubit_error = depolarizing_error(alpha, 2)
            
            noise_model.add_all_qubit_quantum_error(single_qubit_error, 
                                                ['h', 'u1', 'u2', 'u3'])
            noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])
            
            # Hacemos la simulacion
            simulator = AerSimulator(noise_model = noise_model)
            transpiled_circuit = transpile(circuit, simulator)
            result = simulator.run(transpiled_circuit, shots = shots).result()
            counts = result.get_counts()
            
            # Calculamos el QBER
            # Solo consideramos errores si no son 00 o 11
            correct_states = ['00', '11']
            errors = sum([counts.get(state, 0) for state in counts if state not in correct_states])
            qber = errors / shots
            qber_values.append(qber)
        
        # Creamos el gráfico
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, qber_values, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Nivel de ruido (α)', fontsize=12)
        plt.ylabel('QBER', fontsize=12)
        plt.title('Tasa de Error de Bit Cuántico (QBER) vs Nivel de Ruido', fontsize=14)
        plt.grid(True, alpha=0.3)
    
        # Añadimos la línea teórica para comparar los resultados que hemos obtenido
        theoretical_qber = [alpha * 0.75 for alpha in alphas]  # el QUBER teoricamente es 3α/4
        plt.plot(alphas, theoretical_qber, 'r--', label='QBER teórico (3α/4)')
        plt.legend()

        if path:
            plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.show()
        

    


if __name__ == '__main__':

    simulation = Quantum_Simulation()
    circuit = simulation.define_Circuit()
    '''simulation.draw_Circuit(circuit=circuit)

    backend, circuit_isa = simulation.get_Backend_ISA(circuit=circuit)
    simulation.get_Characteristics(circuit_isa=circuit_isa)
    
    job = simulation.execute(backend=backend, circuit_isa=circuit_isa)
    print('\n', job.job_id())'''
    job = simulation.continue_Job(os.getenv('JOB_ID'))

    simulation.plot_Results(job=job)
    # simulation.plot_Results(job=job, probs=True)
    # simulation.plot_Noise_Model(circuit=circuit)
    # simulation.plot_QBER_vs_Alpha(circuit=circuit, alphas=np.linspace(0, 1, 21), shots=5000)