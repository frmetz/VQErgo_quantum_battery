import numpy as np 
import pickle
import matplotlib.pyplot as plt
import itertools
####################################################### Parameters of the systems #######################################################
n = 4           # number of qubits (total cells)
m = 2           # number of cells we want to extract the energy (M <= N)
h  = -0.6       # a transverse magnetic field
J  = -2.0       # J is nearest-neighbor interaction strength
reps = 16        # number of time-step for pvqd method
max_time = 1.6      # evolution time
times = np.linspace(0.0,max_time,reps+1) 
shots = 2048

####################################################### Define variable and directory name #######################################################
dir_name = "results/"
dir_name = dir_name + f"n{n}_J{J}_h{h}/time{max_time}_reps{reps}_shots{shots}/"
dir_name = dir_name + f"m{m}/"
index_reps = np.arange(reps+1)
seeds = np.arange(1,101)
# seeds = np.delete(seeds,3)
work_list = np.zeros((len(seeds),reps+1))
ergo_list = np.zeros((len(seeds),reps+1,1000))
passive_energy_list = np.zeros((len(seeds),reps+1,1000))

####################################################### Load data qiskit #######################################################
for i,index_rep in enumerate(index_reps):
    for j,seed in enumerate(seeds):
        dir_name_time = dir_name + f"time_index{index_rep}/"

        ######## Load data of quantities
        dir_name_quantities = dir_name_time + "quantities/"
        save_name_quantities = dir_name_quantities + f"quantities_seed{seed}"
        quantities = np.load(save_name_quantities+'.npy',allow_pickle=True)
        mean_energy = quantities[0]
        total_work = quantities[1]
        ergotropy = quantities[3]
        work_list[j,i] = total_work
        iteration = int(quantities[4])
        # exec_time = quantities[5]

        ######## Load data of intermediate info: 'nfev', 'parameters', 'energy', 'stddev'
        dir_name_inter_info = dir_name_time + "intermediate_info/"
        save_name_intermediate_info = dir_name_inter_info + f"intermediate_info_seed{seed}"
        with open(save_name_intermediate_info+'.pkl', 'rb') as handle:
            intermediate_info = pickle.load(handle)
            passive_energy = intermediate_info['energy']
        for k in range(iteration):
            passive_energy_list[j,i,k]  = passive_energy[k]
            ergotropy = mean_energy - passive_energy[k]
            ergo_list[j,i,k] = ergotropy

matlab_ergo_compared = []
matlab_work_compared = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}_J{int(-J)}_step{reps}.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_work_compared.append(float(row[1]))
    matlab_ergo_compared.append(float(row[2]))

####################################################### Begin plotting #######################################################
##### Define font, size, type of text, ls, color, markerstyle, etc
plt.rc('font', family='serif')#, serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)
plt.rc('axes', labelsize=25)
plt.rc('legend', fontsize=25)
plt.rc('legend', handlelength=2)
plt.rc('font', size=25)
linestyles = ["-", "--", "dotted"]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#markers = plt.rcParams['axes.prop_cycle'].by_key()['marker']
#colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
#markers = ["o", "v", "s", "d", "*"]
markers = itertools.cycle(("o", "v", "s", "d", "*"))

import matplotlib.cm as mplcm
import matplotlib.colors as colors

NUM_COLORS = 100
cm = plt.get_cmap('gist_rainbow')
cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)

##### Specify columns and rows
fig = plt.figure(figsize=(12,15))
gs = fig.add_gridspec(nrows=2,ncols=1) #height_ratios=[1,1], width_ratios=[1,1])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0])
# ax3 = fig.add_subplot(gs[0,2])
# ax4 = fig.add_subplot(gs[0,3])
# ax5 = fig.add_subplot(gs[1,0])
# ax6 = fig.add_subplot(gs[1,1])
# ax7 = fig.add_subplot(gs[1,2])
# ax8 = fig.add_subplot(gs[1,3])
ax1.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
ax2.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(NUM_COLORS)])

ax1.axhline(y=matlab_ergo_compared[5], color = "black",ls=linestyles[1],lw=3,label="ED")
ax2.axhline(y=matlab_ergo_compared[9], color = "black",ls=linestyles[1],lw=3,label="ED")
# ax3.axhline(y=matlab_ergo_compared[7], color = "black",ls=linestyles[1],lw=3,label="ED")
# ax4.axhline(y=matlab_ergo_compared[8], color = "black",ls=linestyles[1],lw=3,label="ED")
# ax5.axhline(y=matlab_ergo_compared[9], color = "black",ls=linestyles[1],lw=3,label="ED")
# ax6.axhline(y=matlab_ergo_compared[10], color = "black",ls=linestyles[1],lw=3,label="ED")
# ax7.axhline(y=matlab_ergo_compared[11], color = "black",ls=linestyles[1],lw=3,label="ED")
# ax8.axhline(y=matlab_ergo_compared[12], color = "black",ls=linestyles[1],lw=3,label="ED")
##### Plot the average value of data
#for j,passive_seed in enumerate(passive_seeds):
for j in range(0,100):
    ax1.plot(ergo_list[j,5,:], ms=3, marker = next(markers))
    ax2.plot(ergo_list[j,9,:], ms=3, marker = next(markers))
    # ax3.plot(ergo_list[j,7,:], ms=3, marker = "o",ls='dotted')
    # ax4.plot(ergo_list[j,8,:], ms=3, marker = "o",ls='dotted')
    # ax5.plot(ergo_list[j,9,:], ms=3, marker = "o",ls='dotted')
    # ax6.plot(ergo_list[j,10,:], ms=3, marker = "o",ls='dotted')
    # ax7.plot(ergo_list[j,11,:], ms=3, marker = "o",ls='dotted')
    # ax8.plot(ergo_list[j,12,:], ms=3, marker = "o",ls='dotted')

ax1.text(0.01, 0.92,'a)',transform=ax1.transAxes,size=25, weight='bold')
ax2.text(0.01, 0.92,'b)',transform=ax2.transAxes,size=25, weight='bold')
ax1.text(0.04, 0.92,f"t = {np.round(times[5],1)}",transform=ax1.transAxes,size=25, weight='bold')
ax2.text(0.04, 0.92,f"t = {np.round(times[9],1)}",transform=ax2.transAxes,size=25, weight='bold')

##### Set label of x,y axis, label of plots and super title of figure
ax1.set_xlabel(r"Iteration")
ax1.set_ylabel(r"$\mathcal{E}$")
ax1.set_xlim([-10,200])
ax1.set_ylim([-0.5,1.3])

ax2.set_xlabel(r"Iteration")
ax2.set_ylabel(r"$\mathcal{E}$")
ax2.set_xlim([-10,200])
ax2.set_ylim([-1,1.5])
# ax3.set_title(f"t = {np.round(times[7],1)}")

# ax4.set_title(f"t = {np.round(times[8],1)}")

# ax5.set_xlabel(r"Iteration")
# ax5.set_ylabel(r"$\mathcal{E}$")
# ax5.set_title(f"t = {np.round(times[9],1)}")

# ax6.set_xlabel(r"Iteration")
# #ax6.set_ylabel(r"$\mathcal{E}$")
# ax6.set_title(f"t = {np.round(times[10],1)}")

# ax7.set_xlabel(r"Iteration")
# #ax7.set_ylabel(r"$\mathcal{E}$")
# ax7.set_title(f"t = {np.round(times[11],1)}")

# ax8.set_xlabel(r"Iteration")
# #ax8.set_ylabel(r"$\mathcal{E}$")
# ax8.set_title(f"t = {np.round(times[12],1)}")
# fig.suptitle(r"Turning off the magnetic field, $N=%s,M=%s,h=%s,J=%s$, without noise" %(n,m,-h,-J))

##### Close and save figure
ax1.legend()
plt.tight_layout()
plt.savefig(dir_name+f"rxx_noisy_N{n}_M{m}_200iteration.png", dpi=300)
plt.show()
#plt.close()




