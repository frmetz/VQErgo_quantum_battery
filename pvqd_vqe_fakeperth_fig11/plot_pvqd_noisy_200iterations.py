import numpy as np 
import pickle
import matplotlib.pyplot as plt
import itertools
####################################################### Parameters of the systems #######################################################
n = 2           # number of qubits (total cells)
m = 1           # number of cells we want to extract the energy (M <= N)
h  = -0.6       # a transverse magnetic field
J  = -2.0       # J is nearest-neighbor interaction strength

####################################################### Optimum parameters for time evolution using PVQD #######################################################
depth = 1       # depth of the ansatz describing the evolved-state
reps = 14        # number of time-step for pvqd method
seed = 34       # random seed for optimization
time = 1.4      # evolution time
maxiter = 1000
####################################################### Define variable and directory name #######################################################
dir_name = "results/"
dir_name = dir_name + f"n{n}_J{J}_h{h}/"
save_name = dir_name + f"time{time}_depth{depth}_reps{reps}_seed{seed}_maxiter1000_shots1024"
with open(save_name+'.pkl', 'rb') as handle:
    pvqd_result = pickle.load(handle)
times = pvqd_result.times

dir_name = dir_name + f"m{m}/"
index_reps = np.arange(reps+1)
passive_seeds = np.arange(1,101)
work_list = np.zeros((len(passive_seeds),reps+1))
ergo_list = np.zeros((len(passive_seeds),reps+1,maxiter))

####################################################### Load data of matlab #######################################################
##### N2_M1_J4
# total_work_matlab = [0, 0.0473064427761583, 0.181094736152413, 0.378369338172935, 0.605222675363978, 0.822663179019858, 0.993317158972830, 1.08785259627785, 1.09002073284106, 0.999448909186451, 0.831704617194718,0.615619758424676,0.388335014698849,0.188916104910658,0.0516391663146790,9.93694702213777e-05,0.0431553785460441,0.173406723454909,0.368465793210318,0.594805820191349,0.813523463489926]
# ergotropy_matlab = [0,0,0,0,0.0104453507279569,0.445326358039716,0.786634317945660,0.975705192555693,0.980041465682111,0.798897818372902,0.463409234389435,0.0312395168493514,0,0,0,0,0,0,0,0,0.427046926979853]

##### N2_M1_J2
total_work_matlab = [0,0.0471358888288627,0.178471431511003,0.365942495202304,0.569489757586686,0.745618690869516,0.856693589739056,0.878979670344248,0.807714782897564,0.658127000680472,0.462180644289395,0.261746059434094,0.0996526485829374,0.0105369687743537,0.0134414987278692]
ergotropy_matlab = [0,0,0,0,0,0.291237381739033,0.513387179478111,0.557959340688496,0.415429565795129,0.116254001360943,0,0,0,0,0]

####################################################### Load data qiskit #######################################################
for i,index_rep in enumerate(index_reps):
    for j,passive_seed in enumerate(passive_seeds):
        dir_name_time = dir_name + f"time_index{index_rep}/"

        ######## Load data of quantities
        dir_name_quantities = dir_name_time + "quantities/"
        save_name_quantities = dir_name_quantities + f"quantities_seed{passive_seed}"
        quantities = np.load(save_name_quantities+'.npy',allow_pickle=True)
        mean_energy = quantities[0]
        total_work = quantities[1]
        ergotropy = quantities[3]
        work_list[j,i] = total_work
        iteration = int(quantities[4])
        # exec_time = quantities[5]

        ######## Load data of intermediate info: 'nfev', 'parameters', 'energy', 'stddev'
        dir_name_inter_info = dir_name_time + "intermediate_info/"
        save_name_intermediate_info = dir_name_inter_info + f"intermediate_info_seed{passive_seed}"
        with open(save_name_intermediate_info+'.pkl', 'rb') as handle:
            intermediate_info = pickle.load(handle)
            passive_energy = intermediate_info['energy']
        for k in range(iteration):
            ergotropy = mean_energy - passive_energy[k]
            ergo_list[j,i,k] = ergotropy

####################################################### Begin plotting #######################################################
##### Define font, size, type of text, ls, color, markerstyle, etc
plt.rc('font', family='serif')#, serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=35)
plt.rc('ytick', labelsize=35)
plt.rc('axes', labelsize=35)
plt.rc('legend', fontsize=25)
plt.rc('legend', handlelength=2)
plt.rc('font', size=25)
linestyles = ["-", "--", "dotted"]
#colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
# markers = ["o", "v", "s", "d", "*"]
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# markers = itertools.cycle(("o", "v", "s", "d", "*"))


# import matplotlib.cm as mplcm
# import matplotlib.colors as colors

# NUM_COLORS = 100
# cm = plt.get_cmap('gist_rainbow')
# cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
# scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)

##### Specify columns and rows
fig = plt.figure(figsize=(12,14))
gs = fig.add_gridspec(nrows=2,ncols=1) #height_ratios=[1,1], width_ratios=[1,1])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0])

# ax1.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
# ax2.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(NUM_COLORS)])

ax1.axhline(y=ergotropy_matlab[5],color = "black" ,ls=linestyles[1],lw=3,label="ED")
ax2.axhline(y=ergotropy_matlab[9], color="black",ls=linestyles[1],lw=3,label="ED")


average_ergo = np.average(ergo_list,axis=0)
standard_deviation_ergo = np.std(ergo_list,axis=0,ddof=1)
iteration_list = np.arange(1,1001)

ax1.plot(iteration_list,average_ergo[5,:],ls='-',lw=4,color = 'blue',marker = "o")
ax2.plot(iteration_list,average_ergo[9,:],ls='-',lw=4,color = 'blue',marker = "o")
ax1.fill_between(iteration_list, average_ergo[5,:]-standard_deviation_ergo[5,:],average_ergo[5,:]+standard_deviation_ergo[5,:],color='deepskyblue')
ax2.fill_between(iteration_list,average_ergo[9,:] - standard_deviation_ergo[9,:],average_ergo[9,:] + standard_deviation_ergo[9,:],color='deepskyblue')
ax1.text(0.75, 0.1,f'a) t = {np.round(times[5],1)}',transform=ax1.transAxes,size=35, weight='bold')
ax2.text(0.75, 0.1,f'b) t = {np.round(times[9],1)}',transform=ax2.transAxes,size=35, weight='bold')


##### Set label of x,y axis, label of plots and super title of figure
# ax1.set_xlabel(r"Iteration")
ax1.set_ylabel(r"$\mathcal{E}$")
ax1.set_xlim([-10,250])
#ax1.set_ylim([-0.5,1.3])
ax2.set_xlabel(r"Iteration")
ax2.set_ylabel(r"$\mathcal{E}$")
ax2.set_xlim([-10,250])


##### Close and save figure
ax1.legend(loc=8)
ax1.tick_params('both', length=10, width=2, which='major')
ax2.tick_params('both', length=10, width=2, which='major')
plt.tight_layout()
plt.savefig(dir_name+f"pvqd_noisy_N{n}_M{m}_200iteration.eps", dpi=300)
plt.show()
#plt.close()




