from ncpol2sdpa import generate_variables, SdpRelaxation, flatten
import numpy as np

# Define spin-1/2 operators (Pauli matrices divided by 2)
S1 = generate_variables('S1', 3)  # S1[0]=S1x, S1[1]=S1y, S1[2]=S1z
S2 = generate_variables('S2', 3)  # S2[0]=S2x, S2[1]=S2y, S2[2]=S2z

# Heisenberg Hamiltonian (J=1)
H = S1[0]*S2[0] + S1[1]*S2[1] + S1[2]*S2[2]

# Spin-1/2 constraints:
constraints = []
for S in [S1, S2]:
    # Commutation relations: [S^x, S^y] = iS^z (cyclic)
    constraints.append(S[0]*S[1] - S[1]*S[0] - 1j*S[2])  # [Sx,Sy]=iSz
    constraints.append(S[1]*S[2] - S[2]*S[1] - 1j*S[0])  # [Sy,Sz]=iSx
    constraints.append(S[2]*S[0] - S[0]*S[2] - 1j*S[1])  # [Sz,Sx]=iSy

    # Normalization: (S^x)^2 = (S^y)^2 = (S^z)^2 = (1/2)^2 = 0.25
    constraints.append(S[0]**2 - 0.25)
    constraints.append(S[1]**2 - 0.25)
    constraints.append(S[2]**2 - 0.25)

# Initialize SDP relaxation
sdp = SdpRelaxation(flatten([S1, S2]))

# Use higher relaxation level (level=2 or 3) for better accuracy
sdp.get_relaxation(level=2, objective=H, inequalities=[], equalities=constraints)

# Solve with SDPA solver (recommended for stability)
sdp.solve(solver='sdpa')  # Install via: pip install sdpa-python

# Extract the ground state energy
ground_energy = sdp.primal
print(f"Ground state energy: {ground_energy:.6f}")
