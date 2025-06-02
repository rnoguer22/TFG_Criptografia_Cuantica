from qiskit.circuit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_ibm_runtime import SamplerV2

from numpy import pi
from dotenv import load_dotenv
import matplotlib.pyplot as plt



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
        service=QiskitRuntimeService(channel="ibm_quantum")
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
    


if __name__ == '__main__':

    simulation = Quantum_Simulation()
    circuit = simulation.define_Circuit()
    simulation.draw_Circuit(circuit=circuit)

    backend, circuit_isa = simulation.get_Backend_ISA(circuit=circuit)
    simulation.get_Characteristics(circuit_isa=circuit_isa)
    
    job = simulation.execute(backend=backend, circuit_isa=circuit_isa)
    print('\n', job.job_id())
    # job = simulation.continue_Job()

    simulation.plot_Results(job=job)
