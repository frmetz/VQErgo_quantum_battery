import numpy as np 
import pickle
import matplotlib.pyplot as plt
import itertools
####################################################### Parameters of the systems #######################################################
n = 8           # number of qubits (total cells)
m = 7           # number of cells we want to extract the energy (M <= N)
h  = -0.6       # a transverse magnetic field
J  = -2.0       # J is nearest-neighbor interaction strength

####################################################### Optimum parameters for time evolution using PVQD #######################################################
depth = 5       # depth of the ansatz describing the evolved-state
reps = 7        # number of time-step for pvqd method
seed = 35       # random seed for optimization
time = 1.4      # evolution time
####################################################### Define variable and directory name #######################################################
dir_name = "results/"
dir_name = dir_name + f"n{n}_J{J}_h{h}/"
save_name = dir_name + f"time{time}_depth{depth}_reps{reps}_seed{seed}"
with open(save_name+'.pkl', 'rb') as handle:
    pvqd_result = pickle.load(handle)
times = pvqd_result.times

dir_name = dir_name + f"m{m}/"
passive_seeds = np.arange(0,100)
# ergo_list = np.zeros((len(passive_seeds),6000))
# passive_seeds = [52]

dir_name_depth = dir_name + "depth_pass1/"
####################################################### Load data qiskit #######################################################
for j,passive_seed in enumerate(passive_seeds):
    dir_name_time = dir_name_depth + f"time_index{2}/"

    ######## Load data of quantities
    save_name_quantities = dir_name_time+ f"quantities_seed{passive_seed}"
    quantities = np.load(save_name_quantities+'.npy',allow_pickle=True)
    mean_energy = quantities[0]
    iteration = int(quantities[4])
    ergotropy = quantities[3]
    print(ergotropy,passive_seed)
    
    ergo_list = np.zeros((iteration))
    # # ######## Load data of intermediate info: 'nfev', 'parameters', 'energy', 'stddev'
    # dir_name_inter_info = dir_name_time + "intermediate_info/"
    # save_name_intermediate_info = dir_name_inter_info + f"intermediate_info_seed{passive_seed}"
    # with open(save_name_intermediate_info+'.pkl', 'rb') as handle:
    #     intermediate_info = pickle.load(handle)
    #     passive_energy = intermediate_info['energy']
    # for k in range(iteration):
    #     ergotropy = mean_energy - passive_energy[k]
    #     ergo_list[k] = ergotropy

####################################################### Begin plotting #######################################################
##### Define font, size, type of text, ls, color, markerstyle, etc
# plt.rc('font', family='serif')#, serif='Times')
# plt.rc('text', usetex=True)
# plt.rc('xtick', labelsize=25)
# plt.rc('ytick', labelsize=25)
# plt.rc('axes', labelsize=25)
# plt.rc('legend', fontsize=25)
# plt.rc('legend', handlelength=2)
# plt.rc('font', size=25)
# linestyles = ["-", "--", "dotted"]
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# markers = itertools.cycle(("o", "v", "s", "d", "*"))


# import matplotlib.cm as mplcm
# import matplotlib.colors as colors

# NUM_COLORS = 100
# cm = plt.get_cmap('gist_rainbow')
# cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
# scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)

# ##### Specify columns and rows
# fig = plt.figure(figsize=(10,10))
# gs = fig.add_gridspec(nrows=1,ncols=1) #height_ratios=[1,1], width_ratios=[1,1])
# ax1 = fig.add_subplot(gs[0,0])
# ax1.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
# # ax1.axhline(y=ergotropy_matlab[5],color = "black" ,ls=linestyles[1],lw=3,label="ED")


# # for j in range(0,100):
# ax1.plot(ergo_list, ms=3, marker = next(markers))
    

# ##### Set label of x,y axis, label of plots and super title of figure
# ax1.set_xlabel(r"Iteration")
# ax1.set_ylabel(r"$\mathcal{E}$")
# # ax1.set_ylim([0.8,1.2])

# plt.tight_layout()
# plt.show()




