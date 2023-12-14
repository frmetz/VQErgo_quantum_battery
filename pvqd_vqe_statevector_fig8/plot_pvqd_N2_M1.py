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

####################################################### Define variable and directory name #######################################################
dir_name = "results/"
dir_name = dir_name + f"n{n}_J{J}_h{h}/"
save_name = dir_name + f"time{time}_depth{depth}_reps{reps}_seed{seed}"
with open(save_name+'.pkl', 'rb') as handle:
    pvqd_result = pickle.load(handle)
times = pvqd_result.times
dir_name = dir_name + f"m{m}/"
index_reps = np.arange(reps+1)
passive_seeds = np.arange(1,101)
work_list = np.zeros((len(passive_seeds),reps+1))
ergo_list = np.zeros((len(passive_seeds),reps+1))
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
        ergo_list[j,i] = ergotropy
average_ergo = np.average(ergo_list,axis=0)
average_work = np.average(work_list,axis=0)
standard_deviation_ergo = np.std(ergo_list,axis=0,ddof=1)
standard_deviation_work = np.std(work_list,axis=0,ddof=1)


dir_name_new = "results_noise/"
dir_name = dir_name_new + f"n{n}_J{J}_h{h}/"
save_name = dir_name + f"time{time}_depth{depth}_reps{reps}_seed{seed}_maxiter1000_shots1024"
with open(save_name+'.pkl', 'rb') as handle:
    pvqd_result = pickle.load(handle)
times = pvqd_result.times
dir_name = dir_name + f"m{m}/"
index_reps = np.arange(reps+1)
passive_seeds = np.arange(1,101)
work_list = np.zeros((len(passive_seeds),reps+1))
ergo_list = np.zeros((len(passive_seeds),reps+1))
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
        ergo_list[j,i] = ergotropy
average_ergo_noise = np.average(ergo_list,axis=0)
average_work_noise = np.average(work_list,axis=0)
standard_deviation_ergo_noise = np.std(ergo_list,axis=0,ddof=1)
standard_deviation_work_noise = np.std(work_list,axis=0,ddof=1)


matlab_ergo = []
matlab_time = []
matlab_work = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}_J2.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time.append(float(row[0]))
    matlab_work.append(float(row[1]))
    matlab_ergo.append(float(row[2]))
####################################################### Begin plotting #######################################################
##### Define font, size, type of text, ls, color, markerstyle, etc
plt.rc('font', family='serif')#, serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
plt.rc('axes', labelsize=30)
plt.rc('legend', fontsize=25)
plt.rc('legend', handlelength=2)
plt.rc('font', size=25)
linestyles = ["-", "--", "dotted"]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
markers = ["o", "v", "s", "d", "*",'^','v','>','<','p','h']
colors = ['#00429d', '#5e952d', '#ffc35a', '#8c5085', '#e9002c']
##### Specify columns and rows
fig = plt.figure(figsize=(12,10))
gs = fig.add_gridspec(nrows=1,ncols=1) #height_ratios=[1,1], width_ratios=[1,1])
ax1 = fig.add_subplot(gs[0,0])



ax1.plot(matlab_time, matlab_work, ls=linestyles[1],lw=3,color = 'dimgrey',label=r"$W$-ED")
ax1.plot(times, average_work,ms=10, marker = "^",ls='',label=r"$W$-statevector",color = colors[4])#,color = 'blue')
ax1.errorbar(times, average_work_noise, yerr=standard_deviation_work_noise,capsize =6 ,ms=8, marker = "v",ls='',label=r"$W$-FakePerth",color = 'limegreen')
ax1.plot(matlab_time, matlab_ergo, ls=linestyles[0],lw=3,color = 'orange',label=r"$\mathcal{E}$-ED")
ax1.plot(times, average_ergo ,ms=8, marker = "s",ls='',label=r"$\mathcal{E}$-statevector",color = colors[3])#,color = 'green')
ax1.errorbar(times, average_ergo_noise, yerr=standard_deviation_ergo_noise,capsize =6 ,ms=8, marker = "o",ls='',label=r"$\mathcal{E}$-FakePerth",color = 'dodgerblue')
ax1.text(0.05, 0.9,r"$N=%s,M=%s$" %(n,m),transform=ax1.transAxes,size=30, weight='bold')


##### Set label of x,y axis, label of plots and super title of figure
ax1.set_xlabel(r"$t$")
ax1.set_ylabel("Energy")
# ax1.set_xticks([0.0, 0.4, 0.8, 1.2])
# fig.suptitle(r"$N=%s,M=%s$" %(n,m))

##### Close and save figure
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,2,1,3,4,5]
ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order]) 
ax1.tick_params('both', length=10, width=2, which='major')
plt.tight_layout()
plt.savefig(dir_name+f"pvqd_N{n}_M{m}_ver2.eps", dpi=300)
plt.show()
#plt.close()




