##########################################################################################################################################################
##################################### Ising chain Hamiltonian: https://doi.org/10.1103/PhysRevB.100.115142 ###############################################
##########################################################################################################################################################



############################################################# Import packages ############################################################################
import os
import sys
import itertools
import pickle
import numpy as np
#import matplotlib.pyplot as plt
#from scipy.optimize import minimize
#from numpy.random import default_rng
import timeit
import qiskit

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.evolution import expm_multiply_parallel
##########################################################################################################################################################

dir_name = "results/"

# ftol, maxiter, optimizer, ansatz, (time)
repss = [2,7,14,28]
depths = np.arange(1,6)
seeds = np.arange(3) + 33


a = [repss, depths, seeds]
b = list(itertools.product(*a))
print(len(b))

if len(sys.argv) > 1:
    job_id = int(sys.argv[1])
else:
    job_id = 0
c = b[job_id]
print(job_id)
print(c)

reps = c[0]
depth = c[1]
seed = c[2]

n = 6
h = -0.6
J = -2.0
time = 1.4
dt = time/reps
print(dt)

dir_name = dir_name + f"n{n}_J{J}_h{h}/"
os.makedirs(dir_name, exist_ok=True)
save_name = dir_name + f"time{time}_depth{depth}_reps{reps}_seed{seed}"

########################################
# time evolution (exact) quspin
########################################

basis = spin_basis_1d(L=n, pblock=None, zblock=None) #, pauli=False)
dtype_cmplx = np.result_type(np.float64, np.complex128)

z_field = [[h,i] for i in range(n)]
J_xx = [[J,i,(i+1)] for i in range(n-1)]
#J_xx.append([J,n-1,0]) # periodic boundary conditions

static1 = [["z",z_field]]
static2 = [["xx",J_xx]]
dynamic = []
H_full = hamiltonian(static1 + static2, dynamic, dtype=np.float64, basis=basis, check_herm=False, check_symm=False)
expH_full = expm_multiply_parallel(H_full.tocsr(),a=-1j*time,dtype=dtype_cmplx)
psi_exact = np.zeros(2**n, dtype=np.complex128)
psi_exact[0] = 1.
work_array = np.zeros((2*len(psi_exact),), dtype=psi_exact.dtype)
expH_full.dot(psi_exact, work_array=work_array, overwrite_v=True)


########################################
# time evolution pvq qiskit
########################################
from qiskit.providers.aer import  StatevectorSimulator
from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import Z, X, I  # Pauli Z, X matrices and identity
from qiskit.primitives import Estimator, Sampler
from qiskit.algorithms import TimeEvolutionProblem, PVQD
from qiskit.synthesis.evolution.suzuki_trotter import SuzukiTrotter
from qiskit.algorithms.state_fidelities import ComputeUncompute

qiskit.utils.algorithm_globals.random_seed = seed

backend = StatevectorSimulator()
sampler = Sampler()
fidelity = ComputeUncompute(sampler)
estimator = Estimator()

# Ising Hamiltonian
def H_ising(n):
    # Interactions (I is the identity matrix; X and Y are Pauli matricies; ^ is a tensor product)
    XXs = (X^X^(I^(n-2)))
    Zs = (Z^(I^(n-1)))
    for k in range(1,n-1):
        XXs += (((I^k)^(X^X)^(I^(n-2-k))))
        Zs += (((I^k)^(Z)^(I^(n-1-k))))
    Zs += ((I^(n-1))^Z)
    #XXs += (X^(I^(n-2))^X)  #periodic boundary condition
    H = J*XXs + h*Zs
    return H

H = H_ising(n)


# observable = Pauli("ZZ") # in case you want to measure some observable along the way, e.g. mean energy
ansatz = EfficientSU2(n, su2_gates=['ry', 'rz', 'ry'], entanglement='linear', reps=depth)
initial_parameters = np.zeros(ansatz.num_parameters)
optimizer = L_BFGS_B(maxiter=8000, ftol=1e-6)

# setup the algorithm
pvqd = PVQD(
    fidelity,
    ansatz,
    initial_parameters,
    estimator,
    num_timesteps=reps,
    optimizer=optimizer,
    evolution=SuzukiTrotter(order=2, reps=1)
)

# specify the evolution problem
problem = TimeEvolutionProblem(H, time)#, aux_operators=[H, observable])

# and evolve!
print("evolving")
start_time = timeit.default_timer()
result_pvqd = pvqd.evolve(problem)
print(result_pvqd)
exec_time = timeit.default_timer() - start_time
print("\nDONE: --- {:.4f} seconds ---".format(exec_time))

with open(save_name+'.pkl', 'wb') as handle:
    pickle.dump(result_pvqd, handle, protocol=pickle.HIGHEST_PROTOCOL)
