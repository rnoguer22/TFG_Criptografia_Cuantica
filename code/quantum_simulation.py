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
    def define_Circuit(self, psi_state: QuantumCircuit = None):

        # Definimos los registros cuanticos y clasicos
        qreg = QuantumRegister(3, 'q')
        creg_alice = ClassicalRegister(2, 'ca') 
        creg_bob_x = ClassicalRegister(1, 'cx')

        #Creamos el circuito
        circuit = QuantumCircuit(qreg, creg_alice, creg_bob_x)

        # Preparamos el estado que vamos a teletransportar, el cual sera |+>
        if psi_state is None:
            circuit.h(qreg[2]) # Creamos |+> si no indiciamos ningun estado
        else:
            if psi_state.num_qubits == 1:
                circuit.compose(psi_state, [qreg[2]], inplace=True)
            else:
                raise ValueError("psi_state debe ser un QuantumCircuit de 1 qubit.")

        # Creamos el par de Bell entrelazado |Φ+⟩ entre el qubit q[1] de Alice y q[0] de Bob
        circuit.h(qreg[1])
        circuit.cx(qreg[1], qreg[0])

        # Aplicamos una CNOT sobre q[2] y q[1] (q[2] es el qubit de control, y q[1] el objetivo)
        circuit.cx(qreg[2], qreg[1])
        
        # Aplicamos una puerta de Hadamard sobre q[2]
        circuit.h(qreg[2])

        # Medimos los qubits de Alice
        circuit.measure(qreg[1], creg_alice[0]) # Mide q[1] y guarda en ca[0]
        circuit.measure(qreg[2], creg_alice[1]) # Mide q[2] y guarda en ca[1]

        # Ahora Bob aplica correcciones en q[0] basadas en los resultados clásicos de Alice
        # ca[0] controla la puerta X
        with circuit.if_test((creg_alice[0], 1)):
            circuit.x(qreg[0]) # Se aplica X a q[0] (qubit de Bob)

        # ca[1] controla la puerta Z
        with circuit.if_test((creg_alice[1], 1)):
            circuit.z(qreg[0]) # Aplicamos Z a q[0] (qubit de Bob)

        circuit.barrier()
        # Aplicamos una puerta de Hadamard a q[0] para medir sobre la base X
        circuit.h(qreg[0])
        # Medimos el qubit de Bob, dando como resultado el estado teleportado
        circuit.measure(qreg[0], creg_bob_x[0])

        return circuit, creg_alice, creg_bob_x
    

    # Metodo para dibujar y guardar el circuito
    def draw_Circuit(self, circuit, path: str = None):
        circuit.draw('mpl')
        if path is not None:
            plt.savefig(path)
        plt.show()

    
    # Obtenemos el backend y el ISA, esto es para ejecutar con un procesador cuantico de IBM Quantum
    def get_Backend_ISA(self, circuit):
        service = QiskitRuntimeService(channel="ibm_quantum")
        # El backend sera el procesador menos ocupado en el momento
        backend = service.least_busy(min_num_qubits=100, operational=True)
        # pass_manager
        pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
        # circuito ISA (Instruction Set Architecture)
        circuit_isa = pm.run(circuit)

        return backend, circuit_isa


    # Con este metodo obtenemos un breve resumen de las caracteristicas del circuito
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
        pub = (circuit_isa)
        job = sampler.run([pub])
        
        return job
    

    # Metodo para seguir con un job que se ha quedado pendiente, para no gastar minutos de procesamiento de IBM Quantum
    def continue_Job(self, job_id):
        service = QiskitRuntimeService(channel="ibm_quantum")
        job = service.job(job_id)
        return job
    

    # Metodo para mostrar los resultados
    def plot_Results(self, job, probs: bool = False, path: str = None, creg_bob_x_name: str = 'cx'):

        # Obtenemos el resultado de las medicicones del circuito con IBM Quantum
        resultado = job.result()
        
        # Obtenemos unicamente el conteo de resultados del registro de Bob para ver si se ha hecho la teleportacion correctamente
        bob_raw_counts_original = resultado[0].data[creg_bob_x_name].get_counts()            
        title_suffix = " (Base X)"
        bob_raw_counts = {}
        # Mapeamos los resultados a la base X, ya que hemos medido en la base Z
        # Como hemos aplicado una puerta H justo antes de la medicion del qubit, por las propiedades de la puerta de Hadamard sobre + y - obtenemos que
        for outcome, count in bob_raw_counts_original.items():
            if outcome == '0':
                # Si el resultado al medir en la base Z es 0, significa que antes de aplicar H el qubit se encontraba en +
                bob_raw_counts['+'] = bob_raw_counts.get('+', 0) + count
            elif outcome == '1':
                # Si el resultado es 1, el resultado antes de aplicar la puerta H es -
                bob_raw_counts['-'] = bob_raw_counts.get('-', 0) + count

        # Si indicamos que queremos un grafico con las probabilidades, las calculamos y ploteamos el grafico
        # Si no, mostramos el conteo de los resultados de las mediciones, en lugar de una probabilidad
        if probs:
            shots = sum(bob_raw_counts.values())
            distribucion_probas = {key: val/shots for key, val in bob_raw_counts.items()}
            fig = plot_histogram(distribucion_probas, title = f'Probabilidades del Qubit Teletransportado{title_suffix}', figsize=(5, 5)) 
        else:
            fig = plot_histogram(bob_raw_counts, title = f'Resultados del Qubit Teletransportado{title_suffix}', figsize=(5, 5))

        # Con el codigo de arriba ya obtendriamos el grafico
        # Lo que sigue es para modificar algunos parametros del grafico (grosor de barras, agrandar la fuente, etc.)

        ax = fig.gca() # Obtenemos los ejes del grafico

        # Ajustamos el grosor de las barras del plot
        for patch in ax.patches: # ax.patches es lo que forma las barras
            current_width = patch.get_width()
            patch.set_width(current_width * 0.5)
            # Centramos las barras después de haber cambiado su ancho
            x = patch.get_x()
            patch.set_x(x + (current_width - (current_width * 0.5)) / 2)

        # Ajustamos las etiquetas del eje X (las rotamos y ponemos la fuente en 14 para que se vea mejor)
        for tick in ax.get_xticklabels():
                tick.set_rotation(0) 
                tick.set_fontsize(14) 

        fig.tight_layout()
                
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

    # Definimos el estado que queremos hacer la teleportacion. En este caso sera el estado |+>
    psi_to_teleport = QuantumCircuit(1)
    psi_to_teleport.h(0) 

    circuit, creg, creg_bob_x = simulation.define_Circuit(psi_state=psi_to_teleport)
    simulation.draw_Circuit(circuit=circuit, path='img/Simulation/teleportation/tp_circuit.png')

    # Estas lineas comentadas son para ejecutar utilizando IBM Quantum
    # Si ya hemos ejecutado IBM Quantum, cogemos el id del proceso y asi no gastamos minutos
    '''backend, circuit_isa = simulation.get_Backend_ISA(circuit=circuit)
    simulation.get_Characteristics(circuit_isa=circuit_isa)
    
    job = simulation.execute(backend=backend, circuit_isa=circuit_isa)
    print('\n', job.job_id())'''

    # Obtenemos el id de la simulacion, para no gastar minutos volviendo a ejeuctar, los datos se guardan tras la primera ejecucion 
    job = simulation.continue_Job(os.getenv('TP_JOB_ID'))

    # simulation.plot_Results(job=job, path='img/Simulation/teleportation/results.png')
    simulation.plot_Results(job=job, probs=False, path='img/Simulation/teleportation/tp_results_X_basis.png', 
                     measure_bob_in_x_basis=True, creg_bob_x_name=creg_bob_x.name)
    # simulation.plot_Results(job=job, probs=True)
    # simulation.plot_Noise_Model(circuit=circuit)
    # simulation.plot_QBER_vs_Alpha(circuit=circuit, alphas=np.linspace(0, 1, 21), shots=5000)