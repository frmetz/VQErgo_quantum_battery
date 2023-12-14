############################################################# Import packages ############################################################################
import os
import sys
import itertools
import numpy as np
import timeit
import qiskit
import pickle
#from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import L_BFGS_B, SPSA
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import Z, X, I  # Pauli Z, X matrices and identity
from qiskit.primitives import Estimator

##########################################################################################################################################################

############################################################# Input parameters ###########################################################################
n = 2           # number of qubits (total cells)
m = 1           # number of cells we want to extract the energy (M <= N)
depth_pass = 2  # depth of the circuit to find the passive energy
h  = -0.6       # a transverse magnetic field
J  = -4.0       # J is nearest-neighbor interaction strength

#### optimum parameters for pvqd
seed = 35
reps = 20
depth = 1
time = 1.0      # time for evolving 

#### this code runs each time point 100 times with 100 different random seeds
#index_reps = np.arange(reps+1)
#passive_seeds = np.arange(1,101)

index_reps = [20]
passive_seeds = [20]

a = [index_reps, passive_seeds]
b = list(itertools.product(*a))
print(len(b))

if len(sys.argv) > 1:
    job_id = int(sys.argv[1])
else:
    job_id = 0
c = b[job_id]
print(job_id)
print(c)

index_rep = c[0]
passive_seed = c[1]

dir_name = "results/"
dir_name = dir_name + f"n{n}_J{J}_h{h}/"
os.makedirs(dir_name, exist_ok=True)
save_name = dir_name + f"time{time}_depth{depth}_reps{reps}_seed{seed}"

##########################
estimator = Estimator()  # for calculating expectation values
qiskit.utils.algorithm_globals.random_seed = passive_seed
np.random.seed(passive_seed)
############################################################# Prepare Hamiltonian ########################################################################
######## Ising Hamiltonian ##########
def H_ising(n):
    XXs = X ^ X ^ (I ^ (n - 2))
    Zs  = Z ^ (I ^ (n - 1))
    for k in range(1, n - 1):
        XXs += (I ^ k) ^ (X ^ X) ^ (I ^ (n - 2 - k))
        Zs  += (I ^ k) ^ (Z) ^ (I ^ (n - 1 - k))
    Zs   += (I ^ (n - 1)) ^ Z             
    # XXs  += X ^ (I ^ (n - 2)) ^ X      # periodic boundary condition
    H = J * XXs + h * Zs 
    return H

########### Non-interacting Ising Hamiltonian ##########
def H_0(n, m):
    Zs = (I ^ (n - 1)) ^ Z
    Xs = Z ^ (I ^ (n - 1)) #just to make sure that the Hamiltonian is the sum of Pauli strings
    for k in range(n-m, n - 1):
        Zs += (I ^ k) ^ (Z) ^ (I ^ (n - 1 - k))
    
    H = h * Zs + 0.0 * Xs
    return H

H  = H_ising(n)
H0 = H_0(n, m)
print(H0)
##########################################################################################################################################################

################################# Extract information and calculate ergotropy #################################
with open(save_name+'.pkl', 'rb') as handle:
    pvqd_result = pickle.load(handle)

l               = index_rep
ansatz          = EfficientSU2(n, su2_gates=['ry', 'rz', 'ry'], entanglement='linear', reps=depth)
ansatz          = ansatz.bind_parameters(pvqd_result.parameters[l])
job             = estimator.run(ansatz,H0)
mean_energy     = job.result().values[0]
ansatz_passive  = EfficientSU2(m, reps=depth_pass, entanglement='linear', su2_gates=['rx', 'ry', 'rx'])
full_circuit    = ansatz.compose(ansatz_passive).copy()
initial_point   = np.random.random(full_circuit.num_parameters)*1e-1
intermediate_info = {
    'nfev': [],
    'parameters': [],
    'energy': [],
    'stddev': []
}

def callback(nfev, parameters, energy, stddev):
    intermediate_info['nfev'].append(nfev)
    intermediate_info['parameters'].append(parameters)
    intermediate_info['energy'].append(energy)
    intermediate_info['stddev'].append(stddev)

optimizer_passive    = L_BFGS_B(maxiter=8000, ftol=1e-6)
# optimizer_passive    = SPSA(maxiter=300)
start_time           = timeit.default_timer()
vqe                  = VQE(estimator = estimator, ansatz= full_circuit, optimizer = optimizer_passive,initial_point = initial_point, callback=callback)
result2              = vqe.compute_minimum_eigenvalue(H0)
passive_energy       = result2.optimal_value
iteration            = len(intermediate_info['energy'])
exec_time            = timeit.default_timer() - start_time
ergotropy            = mean_energy - passive_energy
total_work           = mean_energy - m*h

#################################################### Print and save #########################################################################################
print("mean_energy = ", mean_energy)
print("total work: ", total_work)
print("passive_energy = ", passive_energy)
print("ergotropy = ", ergotropy)
print('number of iterations:',len(intermediate_info['energy']))

dir_name = dir_name + f"m{m}/"
dir_name = dir_name + f"time_index{index_rep}/"
dir_name_vqe = dir_name + "vqe/"
dir_name_inter_info = dir_name + "intermediate_info/"
dir_name_quantities = dir_name + "quantities/"


os.makedirs(dir_name_vqe, exist_ok=True)
os.makedirs(dir_name_inter_info, exist_ok=True)
os.makedirs(dir_name_quantities, exist_ok=True)

save_name_vqe = dir_name_vqe + f"vqe_result_seed{passive_seed}"
with open(save_name_vqe+'.pkl', 'wb') as handle:
    pickle.dump(result2, handle, protocol=pickle.HIGHEST_PROTOCOL)

save_name_intermediate_info = dir_name_inter_info + f"intermediate_info_seed{passive_seed}"
with open(save_name_intermediate_info+'.pkl', 'wb') as handle:
    pickle.dump(intermediate_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

save_name_quantities = dir_name_quantities + f"quantities_seed{passive_seed}"
np.save(save_name_quantities+'.npy', [mean_energy,total_work,passive_energy,ergotropy,iteration,exec_time])
