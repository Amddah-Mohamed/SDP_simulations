import numpy as np
import ncpol2sdpa as nc
from ncpol2sdpa import generate_operators, SdpRelaxation, flatten
from sympy.physics.quantum.dagger import Dagger
# Parameters
t = 2.8  # Hopping energy (eV)
N = 10   # Number of unit cells (small for demonstration)

d = 1  # Å
b1 = (2*np.pi/(3*d)) * np.array([1, 1/np.sqrt(3)])
b2 = (2*np.pi/(3*d)) * np.array([1, -1/np.sqrt(3)])
l=1j
BZ=[] #Lattice momentum vectors in the first Brillouin Zone
for p1 in range(N) :
    for p2 in range(N):
        BZ+= [(p1/N)*b1 + (p2/N)*b2]
d1=(d/2)*np.array([1, np.sqrt(3)])
d2=(d/2)*np.array([1, -np.sqrt(3)])
d3=d*np.array([1,0])
D=[d1,d2,d3]

# Create annihilation operators for each sublattice
a = generate_operators('a', len(BZ))          
b = generate_operators('b', len(BZ))
a_dag=generate_operators('a_dag', len(BZ))
b_dag=generate_operators('b_dag', len(BZ))
H = 0
# Nearest-neighbor hopping (A <-> B within and between unit cells)
for i in range(len(BZ)):
    for r in D:
    # Intra-cell hopping (A_i to B_i)
        H -= (2*t/(N**2)) * (np.exp(l*(BZ[i]@r))*(Dagger(a[i]) * b[i]) + np.exp(-l*(BZ[i]@r))*Dagger(b[i]) * a[i])
constraints={}

constraints ={**nc.fermionic_constraints(a), **nc.fermionic_constraints(b)}#adding fermionic anticommutation relations for each sublattice
#constraints = [k - v for k, v in constraints.items()]
for ai in a:
    for bj in b:
        constraints[ai*Dagger(bj)] = -Dagger(bj)*ai
        constraints[Dagger(ai)*bj] = -bj*Dagger(ai)
        constraints[ai*bj] = -bj*ai
        constraints[Dagger(ai) * Dagger(bj)] = - Dagger(bj) * Dagger(ai)   

# Flatten operators and initialize SDP
ops = flatten([a,b])
sdp = SdpRelaxation(ops,parallel=True )
sdp.get_relaxation(level=2, objective=H, substitutions=constraints)
# Solve with SDPA solver (recommended)
sdp.solve(solver='sdpa')
print(f"Solver status: {sdp.status}")
print(f"Ground state energy: {sdp.primal:.4f} eV")