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
        print(distribucion)
        maximo = max(distribucion.items(), key=lambda x: x[1])
        print("Maximo :", maximo)

        if probs:
            # Ploteamos la distribución de probabilidades obtenida
            shots = sum(distribucion.values())
            distribucion_probas = {key: val/shots for key, val in distribucion.items()}
            plot_histogram(distribucion_probas, title= 'Distribución de Probabilidades') 
            plt.show()
        else:
            plot_histogram(distribucion, title= 'Distribución de los Resultados')
            plt.show()
        
        if path is not None:
            plt.savefig(path)
    

    '''def plot_Noise_Model(self, job, alphas: list = [0, 0.5, 1], path: str = None):
        def create_noise_model(alpha):
            noise_model = NoiseModel()
            # Ejemplo: Error de depolarización (ajusta según tus necesidades)
            error = depolarizing_error(alpha, 1)  # alpha = probabilidad de error para 1 qubit
            noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3', 'cx'])  # Puertas afectadas
            return noise_model
    
        counts_list = []
        for alpha in alphas:
            # Configuramos el simulator con el ruido en especifico
            noise_model = create_noise_model(alpha)
            simulator = AerSimulator(noise_model = noise_model)
            
            # Transpilamos y ejecutamos
            print(job.inputs)
            transpiled_circuit = transpile(job.inputs['pubs'][0][0], simulator)
            result = simulator.run(transpiled_circuit, shots = 1000).result()
            counts = result.get_counts()
            counts_list.append(counts)
        
        plot_histogram(counts_list, legend = ['α=0', 'α=0.5', 'α=1'], title="Frecuencias de medición vs. nivel de ruido")
        if path is not None:
            plt.savefig(path)
        plt.show()'''

    
    def plot_Noise_Model(self, circuit: QuantumCircuit, alphas: list = [0, 0.10, 0.20], shots: int = 4096, path: str = None):
        """
        Simulates the given circuit with different noise levels (alpha values)
        and plots the resulting measurement frequencies.

        Args:
            circuit (QuantumCircuit): The quantum circuit to simulate.
            alphas (list): A list of alpha values (error probabilities) for the depolarizing error.
            shots (int): The number of shots for each noisy simulation.
            path (str): Optional path to save the plot. If None, displays the plot.
        """
        # We are now directly receiving the simple 2-qubit circuit
        original_circuit = circuit

        counts_list = []
        legends = []

        print(f"Simulating circuit with {original_circuit.num_qubits} qubits under different noise levels...")

        # --- COLOR SCHEME CHANGE HERE ---
        # Define a colormap to pick colors from
        # 'viridis', 'plasma', 'inferno', 'magma', 'cividis' are good perceptual colormaps
        # 'Blues', 'Greens', 'Reds' are sequential for single hues
        cmap = cm.get_cmap('viridis') # You can change 'viridis' to another colormap
        # Create a list of colors from the colormap, scaled by the number of alphas
        colors = [cmap(i) for i in np.linspace(0, 1, len(alphas))]
        # --- END COLOR SCHEME CHANGE ---

        for alpha in alphas:
            print(f"\n--- Simulating with alpha = {alpha} ---")
            
            noise_model = NoiseModel()
            
            single_qubit_error = depolarizing_error(alpha, 1)
            noise_model.add_all_qubit_quantum_error(single_qubit_error, ['h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg', 'u', 'u1', 'u2', 'u3', 'rx', 'ry', 'rz'])
            
            two_qubit_error = depolarizing_error(alpha, 2) 
            noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx', 'cz', 'iswap', 'rxx', 'ryy', 'rzz', 'ecr'])
            
            simulator = AerSimulator(noise_model=noise_model)
            
            try:
                # Transpile the original 2-qubit circuit for the AerSimulator
                transpiled_circuit = transpile(original_circuit, backend=simulator, optimization_level=0)

                result = simulator.run(transpiled_circuit, shots=shots).result()
                counts = result.get_counts(0) 
                counts_list.append(counts)
                legends.append(f'α={alpha}')
                print(f"Simulation for alpha={alpha} completed. Counts: {counts}")

            except Exception as e:
                print(f"Error simulating for alpha={alpha}: {e}")
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
            print("No simulations completed successfully to plot.")
        

    


if __name__ == '__main__':

    simulation = Quantum_Simulation()
    circuit = simulation.define_Circuit()
    '''simulation.draw_Circuit(circuit=circuit)

    backend, circuit_isa = simulation.get_Backend_ISA(circuit=circuit)
    simulation.get_Characteristics(circuit_isa=circuit_isa)
    
    job = simulation.execute(backend=backend, circuit_isa=circuit_isa)
    print('\n', job.job_id())'''
    job = simulation.continue_Job(os.getenv('JOB_ID'))

    # simulation.plot_Results(job=job)
    simulation.plot_Noise_Model(circuit=circuit)