from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import matplotlib.pyplot as plt

# Crear qubits con nombre personalizado
q = QuantumRegister(3, name='')  # qbit0, qbit1, qbit2
c = ClassicalRegister(3, name='')  # También sin nombre visible
qc = QuantumCircuit(q, c)

qc.qubit_labels = ['0', '1', '1']

# Simulamos que el estado inicial es |111> (no se aplica ninguna puerta X)
# Aplicamos puerta Toffoli (CCX): qbit0 y qbit1 controlan, qbit2 es objetivo
qc.ccx(q[0], q[1], q[2])

# Añadimos una "medición visual"
qc.barrier()
qc.measure(q[0], c[0])
qc.measure(q[1], c[1])
qc.measure(q[2], c[2])

# Dibujar el circuito
fig = qc.draw(output='text')

# Ajustar y guardar sin márgenes
#fig.tight_layout(pad=0)
# fig.subplots_adjust(top=1, bottom=0.5, left=0.5, right=1)
# fig.savefig('./img/Circuits/toffoli_ejemplo.png', bbox_inches='tight', pad_inches=0, dpi=300)

print(fig)