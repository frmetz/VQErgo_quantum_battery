############################################################# Import packages ############################################################################
import os
import sys
import itertools
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
import timeit
import qiskit
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.evolution import expm_multiply_parallel
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
from qiskit.circuit.library import EfficientSU2
from qiskit import transpile

############# Load data ##################
dir_name = "results/"
backend = StatevectorSimulator()
h = -0.6
J = -2.0
time = 1.4

n = 2
repss = [14]
depths_n2 = np.arange(1,6)
seeds_n2 = [34]
dir_name_n = dir_name + f"n{n}_J{J}_h{h}/"
basis = spin_basis_1d(L=n, pblock=None, zblock=None) #, pauli=False)
dtype_cmplx = np.result_type(np.float64, np.complex128)
z_field = [[h,i] for i in range(n)]
J_xx = [[J,i,(i+1)] for i in range(n-1)]
# J_xx.append([J,n-1,0]) # periodic boundary conditions
static1 = [["z",z_field]]
static2 = [["xx",J_xx]]
H_full = hamiltonian(static1 + static2, [], dtype=np.float64, basis=basis, check_herm=False, check_symm=False)
err_list_n2 = np.zeros((len(depths_n2),len(seeds_n2),repss[0]+1))
for i,depth in enumerate(depths_n2):
    for j,reps in enumerate(repss):
        for k,seed in enumerate(seeds_n2):
            save_name = dir_name_n + f"time{time}_depth{depth}_reps{reps}_seed{seed}"
            with open(save_name+'.pkl', 'rb') as handle:
                pvqd_result = pickle.load(handle)
            times_n2 = pvqd_result.times
            for l,time in enumerate(pvqd_result.times):
                #qiskit
                ansatz = EfficientSU2(n, su2_gates=['ry', 'rz', 'ry'], entanglement='linear', reps=depth)
                ansatz = ansatz.bind_parameters(pvqd_result.parameters[l])
                circuit = transpile(ansatz, backend=backend)
                job = backend.run(circuit)
                result = job.result()
                state_pvqd = result.get_statevector(circuit, decimals=17).data
                #quspin
                expH_full = expm_multiply_parallel(H_full.tocsr(),a=-1j*time,dtype=dtype_cmplx)
                psi_exact = np.zeros(2**n, dtype=np.complex128)
                psi_exact[0] = 1.
                work_array = np.zeros((2*len(psi_exact),), dtype=psi_exact.dtype)
                expH_full.dot(psi_exact, work_array=work_array, overwrite_v=True)
                err = 1-np.abs(np.conj(state_pvqd).dot(psi_exact))**2
                err_list_n2[i,k,l] = err


n = 4
repss = [14]
depths_n4 = np.arange(1,6)
seeds_n4 = [34]
dir_name_n = dir_name + f"n{n}_J{J}_h{h}/"
basis = spin_basis_1d(L=n, pblock=None, zblock=None) #, pauli=False)
dtype_cmplx = np.result_type(np.float64, np.complex128)
z_field = [[h,i] for i in range(n)]
J_xx = [[J,i,(i+1)] for i in range(n-1)]
# J_xx.append([J,n-1,0]) # periodic boundary conditions
static1 = [["z",z_field]]
static2 = [["xx",J_xx]]
H_full = hamiltonian(static1 + static2, [], dtype=np.float64, basis=basis, check_herm=False, check_symm=False)
err_list_n4 = np.zeros((len(depths_n4),len(seeds_n4),repss[0]+1))
for i,depth in enumerate(depths_n4):
    for j,reps in enumerate(repss):
        for k,seed in enumerate(seeds_n4):
            save_name = dir_name_n + f"time{time}_depth{depth}_reps{reps}_seed{seed}"
            with open(save_name+'.pkl', 'rb') as handle:
                pvqd_result = pickle.load(handle)
            times_n4 = pvqd_result.times
            for l,time in enumerate(pvqd_result.times):
                #qiskit
                ansatz = EfficientSU2(n, su2_gates=['ry', 'rz', 'ry'], entanglement='linear', reps=depth)
                ansatz = ansatz.bind_parameters(pvqd_result.parameters[l])
                circuit = transpile(ansatz, backend=backend)
                job = backend.run(circuit)
                result = job.result()
                state_pvqd = result.get_statevector(circuit, decimals=17).data
                #quspin
                expH_full = expm_multiply_parallel(H_full.tocsr(),a=-1j*time,dtype=dtype_cmplx)
                psi_exact = np.zeros(2**n, dtype=np.complex128)
                psi_exact[0] = 1.
                work_array = np.zeros((2*len(psi_exact),), dtype=psi_exact.dtype)
                expH_full.dot(psi_exact, work_array=work_array, overwrite_v=True)
                err = 1-np.abs(np.conj(state_pvqd).dot(psi_exact))**2
                err_list_n4[i,k,l] = err


n = 6
repss = [7]
depths_n6 = np.arange(1,6)
seeds_n6 = [35]
dir_name_n = dir_name + f"n{n}_J{J}_h{h}/"
basis = spin_basis_1d(L=n, pblock=None, zblock=None) #, pauli=False)
dtype_cmplx = np.result_type(np.float64, np.complex128)
z_field = [[h,i] for i in range(n)]
J_xx = [[J,i,(i+1)] for i in range(n-1)]
# J_xx.append([J,n-1,0]) # periodic boundary conditions
static1 = [["z",z_field]]
static2 = [["xx",J_xx]]
H_full = hamiltonian(static1 + static2, [], dtype=np.float64, basis=basis, check_herm=False, check_symm=False)
err_list_n6 = np.zeros((len(depths_n6),len(seeds_n6),repss[0]+1))
for i,depth in enumerate(depths_n6):
    for j,reps in enumerate(repss):
        for k,seed in enumerate(seeds_n6):
            save_name = dir_name_n + f"time{time}_depth{depth}_reps{reps}_seed{seed}"
            with open(save_name+'.pkl', 'rb') as handle:
                pvqd_result = pickle.load(handle)
            times_n6 = pvqd_result.times
            for l,time in enumerate(pvqd_result.times):
                #qiskit
                ansatz = EfficientSU2(n, su2_gates=['ry', 'rz', 'ry'], entanglement='linear', reps=depth)
                ansatz = ansatz.bind_parameters(pvqd_result.parameters[l])
                circuit = transpile(ansatz, backend=backend)
                job = backend.run(circuit)
                result = job.result()
                state_pvqd = result.get_statevector(circuit, decimals=17).data
                #quspin
                expH_full = expm_multiply_parallel(H_full.tocsr(),a=-1j*time,dtype=dtype_cmplx)
                psi_exact = np.zeros(2**n, dtype=np.complex128)
                psi_exact[0] = 1.
                work_array = np.zeros((2*len(psi_exact),), dtype=psi_exact.dtype)
                expH_full.dot(psi_exact, work_array=work_array, overwrite_v=True)
                err = 1-np.abs(np.conj(state_pvqd).dot(psi_exact))**2
                err_list_n6[i,k,l] = err



n = 8
repss = [7]
depths_n8 = np.arange(4,9)
seeds_n8 = [35]
dir_name_n = dir_name + f"n{n}_J{J}_h{h}/"
basis = spin_basis_1d(L=n, pblock=None, zblock=None) #, pauli=False)
dtype_cmplx = np.result_type(np.float64, np.complex128)
z_field = [[h,i] for i in range(n)]
J_xx = [[J,i,(i+1)] for i in range(n-1)]
# J_xx.append([J,n-1,0]) # periodic boundary conditions
static1 = [["z",z_field]]
static2 = [["xx",J_xx]]
H_full = hamiltonian(static1 + static2, [], dtype=np.float64, basis=basis, check_herm=False, check_symm=False)
err_list_n8 = np.zeros((len(depths_n8),len(seeds_n8),repss[0]+1))
for i,depth in enumerate(depths_n8):
    for j,reps in enumerate(repss):
        for k,seed in enumerate(seeds_n8):
            save_name = dir_name_n + f"time{time}_depth{depth}_reps{reps}_seed{seed}"
            with open(save_name+'.pkl', 'rb') as handle:
                pvqd_result = pickle.load(handle)
            times_n8 = pvqd_result.times
            for l,time in enumerate(pvqd_result.times):
                #qiskit
                ansatz = EfficientSU2(n, su2_gates=['ry', 'rz', 'ry'], entanglement='linear', reps=depth)
                ansatz = ansatz.bind_parameters(pvqd_result.parameters[l])
                circuit = transpile(ansatz, backend=backend)
                job = backend.run(circuit)
                result = job.result()
                state_pvqd = result.get_statevector(circuit, decimals=17).data
                #quspin
                expH_full = expm_multiply_parallel(H_full.tocsr(),a=-1j*time,dtype=dtype_cmplx)
                psi_exact = np.zeros(2**n, dtype=np.complex128)
                psi_exact[0] = 1.
                work_array = np.zeros((2*len(psi_exact),), dtype=psi_exact.dtype)
                expH_full.dot(psi_exact, work_array=work_array, overwrite_v=True)
                err = 1-np.abs(np.conj(state_pvqd).dot(psi_exact))**2
                err_list_n8[i,k,l] = err

############ Begin plotting #################
plt.rc('font', family='serif')#, serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
plt.rc('axes', labelsize=30)
plt.rc('legend', fontsize=20)
plt.rc('legend', handlelength=2)
plt.rc('font', size=25)
linestyles = ["-", "--", "dotted"]
colors = ['tab:blue', 'tab:red', 'tab:brown','tab:purple', 'tab:green', 'tab:pink','tab:orange', 'tab:gray', 'tab:olive', 'tab:cyan']
markers = ["o", "v", "s", "d", "*","X","^","h"]

fig = plt.figure(figsize=(20,10))
gs = fig.add_gridspec(nrows=2,ncols=2) #height_ratios=[1,1], width_ratios=[1,1])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[1,1])


for i,depth in enumerate(depths_n2):
    for k,seed in enumerate(seeds_n2):
            ax1.plot(times_n2, err_list_n2[i,k,:], marker=markers[i], ls=linestyles[k], c=colors[i], label=f"depth {depth}", ms=6)
for i,depth in enumerate(depths_n4):
    for k,seed in enumerate(seeds_n4):
            ax2.plot(times_n4, err_list_n4[i,k,:], marker=markers[i], ls=linestyles[k], c=colors[i], label=f"depth {depth}", ms=6)
for i,depth in enumerate(depths_n6):
    for k,seed in enumerate(seeds_n6):
            ax3.plot(times_n6, err_list_n6[i,k,:], marker=markers[i], ls=linestyles[k], c=colors[i], label=f"depth {depth}", ms=6)
depths_n8_1 = [4,5]
depths_n8_2 = [6,7,8]
for i,depth in enumerate(depths_n8_1):
    for k,seed in enumerate(seeds_n8):
            ax4.plot(times_n8, err_list_n8[i,k,:], marker=markers[i+3], ls=linestyles[k], c=colors[i+3], ms=6)
for i,depth in enumerate(depths_n8_2):
    for k,seed in enumerate(seeds_n8):
            ax4.plot(times_n8, err_list_n8[i+2,k,:], marker=markers[i+2+3], ls=linestyles[k], c=colors[i+2+3], label=f"depth {depth}", ms=6)

ax1.text(0.03, 0.9,"a) $N=2$",transform=ax1.transAxes,size=30, weight='bold')
ax2.text(0.03, 0.9,'b) $N=4$',transform=ax2.transAxes,size=30, weight='bold')
ax3.text(0.03, 0.9,'c) $N=6$',transform=ax3.transAxes,size=30, weight='bold')
ax4.text(0.03, 0.9,'d) $N=8$',transform=ax4.transAxes,size=30, weight='bold')


ax1.set_xlabel(r"$t$")
ax1.set_ylabel(r"$1-F$")
ax1.set_yscale('log')
ax1.set_xticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])

ax2.set_xlabel(r"$t$")
ax2.set_ylabel(r"$1-F$")
ax2.set_yscale('log')
ax2.set_xticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])

ax3.set_xlabel(r"$t$")
ax3.set_ylabel(r"$1-F$")
ax3.set_yscale('log')
ax3.set_xticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])

ax4.set_xlabel(r"$t$")
ax4.set_ylabel(r"$1-F$")
ax4.set_yscale('log')
ax4.set_xticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])
# fig.suptitle(r"Infidelity between pvqd state and exact state, statevector simulation, $h=%s,J=%s$" %(-h,-J))

ax1.legend()
ax4.legend(loc=6)
plt.tight_layout()
plt.savefig(f"pvqd_infidelity.eps", dpi=300)
plt.show()
# plt.close()
