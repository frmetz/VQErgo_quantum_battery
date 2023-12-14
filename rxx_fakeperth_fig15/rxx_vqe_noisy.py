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
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import Z, X, I  # Pauli Z, X matrices and identity
from qiskit import QuantumCircuit
#from qiskit.circuit.library import rxx 

##########################################################################################################################################################

############################################################# Input parameters ###########################################################################
n = 2           # number of qubits (total cells)
m = 1           # number of cells we want to extract the energy (M <= N)
depth_pass = 1  # depth of the circuit to find the passive energy
h  = -0.6       # a transverse magnetic field
J  = -2.0       # J is nearest-neighbor interaction strength
shots = 1024    # number of shots for sampling
maxiter = 500   # number of maximum iterations
reps = 14
max_time = 1.4 
time = np.linspace(0.0,max_time,reps+1)    
print(time)
theta = 2*J*time
print(theta)
#### this code runs each time point 100 times with 100 different random seeds
# index_reps = np.arange(reps+1)
seeds = np.arange(1,101)
#seeds = [20]
a = [seeds]
b = list(itertools.product(*a))
print(len(b))
if len(sys.argv) > 1:
    job_id = int(sys.argv[1])
else:
    job_id = 0
c = b[job_id]
print(job_id)
print(c)

seed = c[0]
print(seed)
dir_name = "results/"
dir_name = dir_name + f"n{n}_J{J}_h{h}/time{max_time}_reps{reps}_shots{shots}/"
dir_name = dir_name + f"m{m}/"
os.makedirs(dir_name, exist_ok=True)

##########################
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.providers.fake_provider import FakeOslo
from qiskit_aer.noise import NoiseModel
device = FakeOslo()
coupling_map = device.coupling_map
noise_model = NoiseModel.from_backend(device)
noisy_estimator = AerEstimator(
    backend_options={
        "method": "density_matrix",
        "coupling_map": coupling_map,
        "noise_model": noise_model,
    },
    run_options={"seed": seed, "shots": shots},
    transpile_options={"seed_transpiler": seed},
)
estimator = noisy_estimator
qiskit.utils.algorithm_globals.random_seed = seed
np.random.seed(seed)
############################################################# Prepare Hamiltonian ########################################################################
######## Ising Hamiltonian ##########
########### Non-interacting Ising Hamiltonian ##########
def H_0(n, m):
    Zs = (I ^ (n - 1)) ^ Z
    Xs = Z ^ (I ^ (n - 1)) #just to make sure that H_0 is the sum of Pauli strings 
    for k in range(n-m, n - 1):
        Zs += (I ^ k) ^ (Z) ^ (I ^ (n - 1 - k))
    H = h * Zs + 0.0 * Xs
    return H

H0 = H_0(n, m)

##########################################################################################################################################################

################################# Time evolution and calculate ergotropy #################################
for l,dt in enumerate(time):
    ansatz          = QuantumCircuit(n)
    for i in range(n-1):
        ansatz.rxx(theta[l],i,i+1)
    job             = estimator.run(ansatz,H0)
    mean_energy     = job.result().values[0]
    ansatz_passive  = EfficientSU2(m, reps=depth_pass, entanglement='linear', su2_gates=['rx', 'ry', 'rx'])
    full_circuit    = ansatz.compose(ansatz_passive).copy()
    initial_point   = np.random.random(full_circuit.num_parameters)*1e-1
    intermediate_info = {
        'nfev': [],
        'parameters': [],
        'energy': [],
        'stepsize': []
    }

    def callback(nfev, parameters,energy ,stepsize,accepted):
        intermediate_info['nfev'].append(nfev)
        intermediate_info['parameters'].append(parameters)
        intermediate_info['energy'].append(energy)
        intermediate_info['stepsize'].append(stepsize)

    #optimizer_passive    = L_BFGS_B(maxiter=8000, ftol=1e-6)
    optimizer_passive    = SPSA(maxiter=maxiter,callback=callback)
    start_time           = timeit.default_timer()
    vqe                  = VQE(estimator = estimator, ansatz= full_circuit, optimizer = optimizer_passive,initial_point = initial_point)
    result2              = vqe.compute_minimum_eigenvalue(H0)
    passive_energy       = result2.optimal_value
    iteration            = len(intermediate_info['energy'])
    exec_time            = timeit.default_timer() - start_time
    ergotropy            = mean_energy - passive_energy
    total_work           = mean_energy - m*h

    #################################################### Print and save #########################################################################################
    # print("mean_energy = ", mean_energy)
    # print("total work: ", total_work)
    # print("passive_energy = ", passive_energy)
    # print("ergotropy = ", ergotropy)
    # print('number of iterations:',len(intermediate_info['energy']))

    
    dir_name_time = dir_name + f"time_index{l}/"
    dir_name_vqe = dir_name_time + "vqe/"
    dir_name_inter_info = dir_name_time + "intermediate_info/"
    dir_name_quantities = dir_name_time + "quantities/"


    os.makedirs(dir_name_vqe, exist_ok=True)
    os.makedirs(dir_name_inter_info, exist_ok=True)
    os.makedirs(dir_name_quantities, exist_ok=True)

    save_name_vqe = dir_name_vqe + f"vqe_result_seed{seed}"
    with open(save_name_vqe+'.pkl', 'wb') as handle:
        pickle.dump(result2, handle, protocol=pickle.HIGHEST_PROTOCOL)

    save_name_intermediate_info = dir_name_inter_info + f"intermediate_info_seed{seed}"
    with open(save_name_intermediate_info+'.pkl', 'wb') as handle:
        pickle.dump(intermediate_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    save_name_quantities = dir_name_quantities + f"quantities_seed{seed}"
    np.save(save_name_quantities+'.npy', [mean_energy,total_work,passive_energy,ergotropy,iteration,exec_time])
