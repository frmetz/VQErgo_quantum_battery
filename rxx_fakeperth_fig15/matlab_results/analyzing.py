import numpy as np 
import pickle
import matplotlib.pyplot as plt
import itertools
####################################################### Parameters of the systems #######################################################
n = 4            # number of qubits (total cells)
m = 2            # number of cells we want to extract the energy (M <= N)
h  = -0.6        # a transverse magnetic field
J  = -2.0        # J is nearest-neighbor interaction strength
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
seeds = np.delete(seeds,3)
# error_work = np.zeros((len(passive_seeds),reps+1))
# error_ergo = np.zeros((len(passive_seeds),reps+1))
work_list = np.zeros((len(seeds),reps+1))
ergo_list = np.zeros((len(seeds),reps+1))


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
        ergo_list[j,i] = ergotropy
        # passive_energy = quantities[2]
        # iteration = quantities[4]
        # exec_time = quantities[5]

        ######## Relative/Absolute error calculation 
        # error_work[j,i] = abs((total_work - total_work_matlab[i])/total_work_matlab[i])
        # error_ergo[j,i] = abs((ergotropy - ergotropy_matlab[i])/ergotropy_matlab[i])
        # error_work[j,i] = abs((total_work - total_work_matlab[i])/total_work_matlab[i])
        # error_ergo[j,i] = abs((ergotropy - ergotropy_matlab[i])/ergotropy_matlab[i])

        ######## Load data of vqe_optimization: 
        # dir_name_vqe = dir_name_time + "vqe/"
        # save_name_vqe = dir_name_vqe + f"vqe_result_seed{passive_seed}"
        # with open(save_name_vqe+'.pkl', 'rb') as handle:
        #     vqe_result = pickle.load(handle)
        # print(vqe_result)

        ######## Load data of intermediate info: 'nfev', 'parameters', 'energy', 'stddev'
        # dir_name_inter_info = dir_name_time + "intermediate_info/"
        # save_name_intermediate_info = dir_name_inter_info + f"intermediate_info_seed{passive_seed}"
        # with open(save_name_intermediate_info+'.pkl', 'rb') as handle:
        #     intermediate_info = pickle.load(handle)

matlab_ergo_compared = []
matlab_work_compared = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}_J{int(-J)}_step{reps}.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_work_compared.append(float(row[1]))
    matlab_ergo_compared.append(float(row[2]))


matlab_ergo = []
matlab_time = []
matlab_work = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}_J{int(-J)}_step200.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time.append(float(row[0]))
    matlab_work.append(float(row[1]))
    matlab_ergo.append(float(row[2]))
####################################################### Begin plotting #######################################################
##### Define font, size, type of text, ls, color, markerstyle, etc
plt.rc('font', family='serif')#, serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('axes', labelsize=20)
plt.rc('legend', fontsize=20)
plt.rc('legend', handlelength=2)
plt.rc('font', size=20)
linestyles = ["-", "--", "dotted"]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
# markers = ["o", "v", "s", "d", "*"]
marker = itertools.cycle(("o", "v", "s", "d", "*"))

##### Specify columns and rows
fig = plt.figure(figsize=(20,10))
gs = fig.add_gridspec(nrows=2,ncols=3) #height_ratios=[1,1], width_ratios=[1,1])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[1,0])
ax5 = fig.add_subplot(gs[1,1])
ax6 = fig.add_subplot(gs[1,2])

##### Plot the average value of data
average_ergo = np.average(ergo_list,axis=0)
average_work = np.average(work_list,axis=0)

#variance_ergo = np.var(ergo_list,axis=0)
#variance_work = np.var(work_list,axis=0)
standard_deviation_ergo = np.std(ergo_list,axis=0,ddof=1)
standard_deviation_work = np.std(work_list,axis=0,ddof=1)
standard_error_ergo     = standard_deviation_ergo/np.sqrt(len(seeds))
standard_error_work     = standard_deviation_work/np.sqrt(len(seeds))
error_work = abs(average_work - matlab_work_compared)
error_ergo = abs(average_ergo - matlab_ergo_compared)

ax1.plot(matlab_time, matlab_work, ls=linestyles[0],lw=3,color = colors[1],label="ED")
ax1.errorbar(times, average_work, yerr=standard_deviation_work,capsize =6 ,ms=6, marker = "o",ls='',label="VQE - depth = 1",color = colors[0])
ax2.plot(times,standard_deviation_work, ms=6, marker="x",color = colors[3],ls='')
ax3.plot(times, error_work, ms=6, marker="x",color = colors[3],ls='')

ax4.plot(matlab_time, matlab_ergo, ls=linestyles[0],lw=3,color = colors[1])
ax4.errorbar(times, average_ergo, yerr=standard_deviation_ergo,capsize =6 ,ms=6, marker = "o",ls='',color = colors[0])
ax5.plot(times,standard_deviation_ergo, ms=6, marker="x",color = colors[3],ls='')
ax6.plot(times, error_ergo, ms=6, marker="x",color = colors[3],ls='')

ax1.text(0.05, 0.9,'a)',transform=ax1.transAxes,size=20, weight='bold')
ax2.text(0.05, 0.9,'b)',transform=ax2.transAxes,size=20, weight='bold')
ax3.text(0.05, 0.9,'c)',transform=ax3.transAxes,size=20, weight='bold')
ax4.text(0.05, 0.9,'d)',transform=ax4.transAxes,size=20, weight='bold')
ax5.text(0.05, 0.9,'e)',transform=ax5.transAxes,size=20, weight='bold')
ax6.text(0.05, 0.9,'f)',transform=ax6.transAxes,size=20, weight='bold')

##### Plot all 100 runs of data
# for j,passive_seed in enumerate(passive_seeds):
#     plt.plot(times, ergo_list[j,:],marker = next(marker),ls ='')


# for i,depth in enumerate(depths):
#     for k,seed in enumerate(seeds):
#             plt.plot(pvqd_result.times, err_list[i,k,:], marker=markers[i], ls=linestyles[k], c=colors[i], label=f"depth {depth}", ms=3)
# plt.yscale('log')

##### Set label of x,y axis, label of plots and super title of figure
ax1.set_xlabel(r"$t$")
ax1.set_ylabel(r"$W$")

ax2.set_xlabel(r"$t$")
ax2.set_ylabel(r"$\sigma(W)$")
#ax2.set_yscale('log')

ax3.set_xlabel(r"$t$")
ax3.set_ylabel(r"$|\Delta W|$")
#ax3.set_yscale('log')


ax4.set_xlabel(r"$t$")
ax4.set_ylabel(r"$\mathcal{E}$")

ax5.set_xlabel(r"$t$")
ax5.set_ylabel(r"$\sigma (\mathcal{E})$")
#ax5.set_yscale('log')

ax6.set_xlabel(r"$t$")
ax6.set_ylabel(r"$|\Delta \mathcal{E}|$")
#ax6.set_yscale('log')
fig.suptitle(r"Turning off the magnetic field, $N=%s,M=%s,h=%s,J=%s$, with noise" %(n,m,-h,-J))

##### Close and save figure
ax1.legend(loc=8)
# ax2.legend()
# ax3.legend()
# ax4.legend()
plt.tight_layout()
plt.savefig(dir_name+f"rxx_noisy_N{n}_M{m}.eps", dpi=300)
plt.show()
#plt.close()




