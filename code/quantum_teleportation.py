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

# Cargar variables de entorno si están configuradas para QiskitRuntimeService
load_dotenv()


class Quantum_Simulation:

    def __init__(self):
        load_dotenv() # Asegúrate de que las variables de entorno se carguen aquí también

    # Método para definir el circuito de TELEPORTACIÓN CUÁNTICA
    def define_teleportation_circuit(self, psi_state: QuantumCircuit = None):
        # Qubits:
        # q[0]: El qubit del estado |ψ⟩ (arriba en la imagen)
        # q[1]: El primer qubit del par Bell |Φ+⟩ (medio en la imagen)
        # q[2]: El segundo qubit del par Bell |Φ+⟩ (abajo en la imagen)
        qreg = QuantumRegister(3, 'q')
        # Registros clásicos:
        # c[0]: Resultado de la medición de q[1] (línea media de resultados)
        # c[1]: Resultado de la medición de q[0] (línea inferior de resultados)
        # c[2]: Resultado final en el qubit de Bob (línea superior de resultados para |ψ>)
        creg = ClassicalRegister(3, 'c')
        circuit = QuantumCircuit(qreg, creg)

        # --- Parte 1: Preparación del Estado ---
        # 1. Preparar el estado a teletransportar |ψ⟩ en q[0]
        #    La imagen muestra |ψ⟩ en el qubit superior.
        if psi_state is None:
            # Por defecto, si no se especifica, teletransportamos un estado |0> (sin operaciones)
            # o puedes poner un ejemplo como |+> o |1>
            pass # No hacer nada si |psi> es |0>
            # Ejemplo para teletransportar |+>:
            # circuit.h(qreg[0])
            # Ejemplo para teletransportar |1>:
            # circuit.x(qreg[0])
        else:
            if psi_state.num_qubits == 1:
                circuit.compose(psi_state, [qreg[0]], inplace=True)
            else:
                raise ValueError("psi_state debe ser un QuantumCircuit de 1 qubit.")

        # 2. Crear el par Bell entrelazado |Φ+⟩ entre q[1] y q[2]
        #    La imagen muestra |Φ+⟩ en los qubits del medio y abajo.
        circuit.h(qreg[1])
        circuit.cx(qreg[1], qreg[2])
        circuit.barrier() # Separador visual, no funcional

        # --- Parte 2: Codificación y Medición de Alice ---
        # (Estas operaciones se aplican a los qubits de Alice: q[0] y q[1])
        # 3. Alice aplica CNOT entre q[0] (control) y q[1] (objetivo)
        circuit.cx(qreg[0], qreg[1])
        
        # 4. Alice aplica Hadamard a q[0]
        circuit.h(qreg[0])
        circuit.barrier() # Separador visual

        # 5. Alice mide sus dos qubits (q[0] y q[1]) y guarda en registros clásicos
        #    La imagen muestra la medición de q[1] en la línea media y q[0] en la línea inferior.
        #    Aseguramos que c[0] almacene el resultado de q[1] y c[1] almacene el resultado de q[0]
        #    para que coincida con el orden de control en las compuertas X y Z de Bob.
        circuit.measure(qreg[1], creg[0]) # Mide q[1] y guarda en c[0]
        circuit.measure(qreg[0], creg[1]) # Mide q[0] y guarda en c[1]
        circuit.barrier() # Separador visual

        # --- Parte 3: Reconstrucción de Bob ---
        # (Estas operaciones se aplican al qubit de Bob: q[2])
        # 6. Bob aplica correcciones basadas en los resultados clásicos de Alice
        #    - X condicional al resultado de la medición de q[1] (almacenado en c[0])
        #    - Z condicional al resultado de la medición de q[0] (almacenado en c[1])
        
        # Si el resultado de c[0] (medición de q[1]) es 1, aplicar X a q[2]
        with circuit.if_test((creg[0], 1)):
            circuit.x(qreg[2])
            
        # Si el resultado de c[1] (medición de q[0]) es 1, aplicar Z a q[2]
        with circuit.if_test((creg[1], 1)):
            circuit.z(qreg[2])

        # Opcional: Medir el qubit final de Bob para verificar que el estado se teletransportó correctamente
        # La imagen muestra el estado final |ψ⟩ en la línea superior, pero no una medición explícita ahí.
        # Añadir esta medición es útil para la verificación.
        circuit.measure(qreg[2], creg[2])

        return circuit
    

    # Metodo para dibujar y guardar el circuito
    def draw_Circuit(self, circuit, path: str = None):
        # Usar output='mpl' para Matplotlib
        circuit.draw('mpl', idle_wires=False) # idle_wires=False para ocultar líneas de qubits no usados si los hay
        if path is not None:
            plt.savefig(path)
        plt.show()

    # Obtenemos el backend y el ISA (sin cambios, usa IBM Quantum si configuras el servicio)
    def get_Backend_ISA(self, circuit):
        service = QiskitRuntimeService(channel="ibm_quantum")
        # El backend sera el procesador menos ocupado en el momento
        backend = service.least_busy(min_num_qubits=3, operational=True) # Necesitamos al menos 3 qubits
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
        # Puedes ajustar o comentar las opciones de supresión de errores para empezar
        # sampler.options.dynamical_decoupling.enable = True          # Supresión de errores: Dynamical Decoupling
        # sampler.options.dynamical_decoupling.sequence_type = "XY4"
        # sampler.options.twirling.enable_gates = True                # Supresión de errores: Pauli Twirling
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
        resultado = job.result()
        # En teleportación, el resultado final esperado en el qubit de Bob es el estado inicial.
        # Por lo tanto, el clásico de interés es el último (c[2] en este caso).
        # Los resultados de SamplerV2 se acceden a través de data.meas_key.get_counts()
        # Si tienes 3 registros clásicos c0, c1, c2, el resultado será un string de 3 bits 'c2c1c0'
        # o 'c2 c1 c0' si se usan ClassicalRegister.
        # Para verificar el estado teletransportado, nos interesa el resultado del qubit de Bob (q[2]),
        # que se mide en creg[2].
        
        # Obtenemos las cuentas directamente.
        # El formato de la clave de resultado será 'c2 c1 c0'.
        # La forma más fácil es obtener todas las cuentas y luego procesarlas.
        raw_counts = resultado[0].data.c.get_counts()
        
        # Queremos analizar solo el resultado del qubit de Bob (c[2]), que es el primer bit en el string
        bob_results = {}
        for outcome_str, count in raw_counts.items():
            # El outcome_str será algo como '0 0 0', '0 0 1', '1 0 0', etc.
            # c[2] es el primer bit en la cadena, c[1] el segundo, c[0] el tercero.
            bob_final_bit = outcome_str[0] # El primer carácter es el resultado de c[2] (qubit de Bob)
            bob_results[bob_final_bit] = bob_results.get(bob_final_bit, 0) + count

        print('\nResultados del qubit de Bob:', bob_results, '\n')
        maximo = max(bob_results.items(), key=lambda x: x[1])
        print("Resultado más frecuente en Bob (teletransportado):", maximo)

        if probs:
            shots = sum(bob_results.values())
            distribucion_probas = {key: val/shots for key, val in bob_results.items()}
            plot_histogram(distribucion_probas, title = 'Distribución de Probabilidades del Qubit Teletransportado') 
        else:
            plot_histogram(bob_results, title = 'Distribución de los Resultados del Qubit Teletransportado')
        
        if path:
            plt.savefig(path)
        plt.show()

    # Métodos plot_Noise_Model y plot_QBER_vs_Alpha
    # Estos métodos están diseñados para CHSH/QBER, no para teleportación directamente.
    # Podrías adaptarlos para analizar el éxito de la teleportación bajo ruido,
    # pero eso implicaría redefinir lo que significa "error" en el contexto de teleportación.
    # Por ahora, los dejaré como están, pero ten en cuenta que no son directamente aplicables
    # a la verificación de la teleportación sin adaptaciones.
    
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
                # Transpilamos el circuito original de 3 qubits para el AerSimulator
                # Usar circuit.copy() para no modificar el original
                transpiled_circuit = transpile(circuit.copy(), backend=simulator, optimization_level=0)

                result = simulator.run(transpiled_circuit, shots=shots).result()
                counts = result.get_counts(0) # Obtiene las cuentas del primer classical register, ajusta si necesitas todos los bits
                counts_list.append(counts)
                legends.append(f'α={alpha}')
                print(f"Simulation for alpha={alpha} completed. Counts: {counts}")

            except Exception as e:
                print(f"Error en la simulación de alpha = {alpha}: {e}")
                counts_list.append({}) 
                legends.append(f'α={alpha} (Error)')
                continue 

        if counts_list: 
            # Asegúrate de que las claves (ej. '000', '001') se alineen para el histograma
            # Esto puede requerir un procesamiento adicional para asegurar que todas las claves posibles estén presentes
            # en todos los diccionarios de counts_list antes de plot_histogram.
            fig = plot_histogram(counts_list, legend=legends, title="Frecuencias de Medición vs. Nivel de Ruido", color=colors)
            if path is not None:
                plt.savefig(path)
            plt.show()
            plt.close(fig)
        else:
            print("Error en la simulación")

    
    def plot_QBER_vs_Alpha(self, circuit: QuantumCircuit, alphas: list = np.linspace(0, 1, 11), shots: int = 1000, path: str = None):
        # Este método es específico para CHSH donde los estados ideales son '00' y '11'.
        # Para teleportación, el "ideal" depende del estado |psi> que se teletransportó.
        # Por lo tanto, no es directamente aplicable sin una redefinición.
        print("plot_QBER_vs_Alpha no es directamente aplicable a la teleportación sin modificaciones.")
        # ... (restos de la función original, no modificados)
        qber_values = []
        
        for alpha in alphas:
            noise_model = NoiseModel()
            single_qubit_error = depolarizing_error(alpha, 1)
            two_qubit_error = depolarizing_error(alpha, 2)
            
            noise_model.add_all_qubit_quantum_error(single_qubit_error, 
                                                ['h', 'u1', 'u2', 'u3']) # Ajusta estas compuertas si tu circuito usa otras
            noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx']) # Ajusta estas compuertas si tu circuito usa otras
            
            simulator = AerSimulator(noise_model = noise_model)
            # Asegúrate de que el circuito para QBER sea el correcto (con 2 qubits y un par Bell)
            # o adapta este método para el circuito de 3 qubits de teleportación
            transpiled_circuit = transpile(circuit.copy(), simulator)
            result = simulator.run(transpiled_circuit, shots = shots).result()
            counts = result.get_counts()
            
            # Esto es específico para un Bell State (00/11)
            correct_states = ['00', '11'] 
            errors = sum([counts.get(state, 0) for state in counts if state not in correct_states])
            qber = errors / shots
            qber_values.append(qber)
        
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, qber_values, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Nivel de ruido (α)', fontsize=12)
        plt.ylabel('QBER', fontsize=12)
        plt.title('Tasa de Error de Bit Cuántico (QBER) vs Nivel de Ruido', fontsize=14)
        plt.grid(True, alpha=0.3)
    
        theoretical_qber = [alpha * 0.75 for alpha in alphas]
        plt.plot(alphas, theoretical_qber, 'r--', label='QBER teórico (3α/4)')
        plt.legend()

        if path:
            plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.show()


# --- Ejemplo de uso ---
if __name__ == '__main__':
    sim = Quantum_Simulation()

    # Definir el estado que queremos teletransportar
    # Por ejemplo, un estado |+> (Hadamard sobre |0>)
    psi_to_teleport = QuantumCircuit(1)
    psi_to_teleport.h(0) 

    # Crear el circuito de teleportación
    teleportation_circuit = sim.define_teleportation_circuit(psi_state=psi_to_teleport)
    
    # Dibujar el circuito
    sim.draw_Circuit(teleportation_circuit, path='img/Teleportation_Circuit.png')

    # Para ejecutar en un simulador local de Qiskit Aer (más rápido para desarrollo)
    simulator = AerSimulator()
    transpiled_circuit = transpile(teleportation_circuit, simulator)
    job = simulator.run(transpiled_circuit, shots=8192)
    result = job.result()
    
    # Mostrar resultados (se enfoca en el qubit teletransportado de Bob)
    sim.plot_Results(result, probs=False, path='img/Teleportation_Results.png')

    # Si quieres usar el backend real de IBM Quantum:
    # try:
    #     backend, circuit_isa = sim.get_Backend_ISA(teleportation_circuit)
    #     sim.get_Characteristics(circuit_isa)
    #     job_ibm = sim.execute(backend, circuit_isa)
    #     # Esperar a que el job termine si es en hardware real
    #     print("Job ID:", job_ibm.job_id())
    #     # sim.plot_Results(job_ibm.result(), probs=False, path='img/Teleportation_Results_IBM.png')
    # except Exception as e:
    #     print(f"Error al ejecutar en IBM Quantum: {e}")

    # Ejemplo de uso de plot_Noise_Model (adaptado para 3 qubits)
    # Ten en cuenta que la interpretación de los resultados de ruido para teleportación
    # es diferente a CHSH.
    # sim.plot_Noise_Model(teleportation_circuit, alphas=[0, 0.1, 0.2, 0.3], shots=4096, path='img/Teleportation_Noise_Model.png')

    # plot_QBER_vs_Alpha no es directamente aplicable para teleportación sin redefinición.
    # sim.plot_QBER_vs_Alpha(teleportation_circuit) # Descomentar solo si lo adaptas para teleportación'''