import numpy as np 
import pickle
import matplotlib.pyplot as plt
import itertools
####################################################### Parameters of the systems #######################################################
n = 8           # number of qubits (total cells)
h  = -0.6       # a transverse magnetic field
J  = -2.0       # J is nearest-neighbor interaction strength
depth = 5       # depth of the ansatz describing the evolved-state
reps = 7        # number of time-step for pvqd method
seed = 35       # random seed for optimization
time = 1.4      # evolution time
dir_name = "results/"
dir_name = dir_name + f"n{n}_J{J}_h{h}/"
save_name = dir_name + f"time{time}_depth{depth}_reps{reps}_seed{seed}"
with open(save_name+'.pkl', 'rb') as handle:
    pvqd_result = pickle.load(handle)
times = pvqd_result.times
index_reps = np.arange(reps+1)
passive_seeds = np.arange(0,99)



m = 1           
depth_passs = [2]
dir_name_m = dir_name + f"m{m}/"
ergo_list = np.zeros((len(passive_seeds),reps+1))
iteration_list = np.zeros((len(passive_seeds),reps+1))
work_list = np.zeros((len(passive_seeds),reps+1))
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name_m + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            total_work = quantities[1]
            ergotropy = quantities[3]
            iteration = quantities[4]
            ergo_list[j,i] = ergotropy
            iteration_list[j,i] = iteration
            work_list[j,i]      = total_work
ergo_average_1 = np.average(ergo_list,axis=0)
iteration_average_1 = np.average(iteration_list,axis=0)
work_average_1 = np.average(work_list,axis=0)
standard_deviation_ergo_1 = np.std(ergo_list,axis=0,ddof=1)

m = 2           
depth_passs = [2]
dir_name_m = dir_name + f"m{m}/"
ergo_list = np.zeros((len(passive_seeds),reps+1))
iteration_list = np.zeros((len(passive_seeds),reps+1))
work_list = np.zeros((len(passive_seeds),reps+1))
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name_m + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            total_work = quantities[1]
            ergotropy = quantities[3]
            iteration = quantities[4]
            ergo_list[j,i] = ergotropy
            iteration_list[j,i] = iteration
            work_list[j,i]      = total_work
ergo_average_2 = np.average(ergo_list,axis=0)
iteration_average_2 = np.average(iteration_list,axis=0)
work_average_2 = np.average(work_list,axis=0)
standard_deviation_ergo_2 = np.std(ergo_list,axis=0,ddof=1)

m = 3           
depth_passs = [2]
dir_name_m = dir_name + f"m{m}/"
ergo_list = np.zeros((len(passive_seeds),reps+1))
iteration_list = np.zeros((len(passive_seeds),reps+1))
work_list = np.zeros((len(passive_seeds),reps+1))
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name_m + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            total_work = quantities[1]
            ergotropy = quantities[3]
            iteration = quantities[4]
            ergo_list[j,i] = ergotropy
            iteration_list[j,i] = iteration
            work_list[j,i]      = total_work
ergo_average_3 = np.average(ergo_list,axis=0)
iteration_average_3 = np.average(iteration_list,axis=0)
work_average_3 = np.average(work_list,axis=0)
standard_deviation_ergo_3 = np.std(ergo_list,axis=0,ddof=1)


m = 4           
depth_passs = [2]
dir_name_m = dir_name + f"m{m}/"
ergo_list = np.zeros((len(passive_seeds),reps+1))
iteration_list = np.zeros((len(passive_seeds),reps+1))
work_list = np.zeros((len(passive_seeds),reps+1))
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name_m + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            total_work = quantities[1]
            ergotropy = quantities[3]
            iteration = quantities[4]
            ergo_list[j,i] = ergotropy
            iteration_list[j,i] = iteration
            work_list[j,i]      = total_work
ergo_average_4 = np.average(ergo_list,axis=0)
iteration_average_4 = np.average(iteration_list,axis=0)
work_average_4 = np.average(work_list,axis=0)
standard_deviation_ergo_4 = np.std(ergo_list,axis=0,ddof=1)


m = 5           
depth_passs = [2]
dir_name_m = dir_name + f"m{m}/"
ergo_list = np.zeros((len(passive_seeds),reps+1))
iteration_list = np.zeros((len(passive_seeds),reps+1))
work_list = np.zeros((len(passive_seeds),reps+1))
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name_m + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            total_work = quantities[1]
            ergotropy = quantities[3]
            iteration = quantities[4]
            ergo_list[j,i] = ergotropy
            iteration_list[j,i] = iteration
            work_list[j,i]      = total_work
ergo_average_5 = np.average(ergo_list,axis=0)
iteration_average_5 = np.average(iteration_list,axis=0)
work_average_5 = np.average(work_list,axis=0)
standard_deviation_ergo_5 = np.std(ergo_list,axis=0,ddof=1)



m = 6           
depth_passs = [2]
dir_name_m = dir_name + f"m{m}/"
ergo_list = np.zeros((len(passive_seeds),reps+1))
iteration_list = np.zeros((len(passive_seeds),reps+1))
work_list = np.zeros((len(passive_seeds),reps+1))
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name_m + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            total_work = quantities[1]
            ergotropy = quantities[3]
            iteration = quantities[4]
            ergo_list[j,i] = ergotropy
            iteration_list[j,i] = iteration
            work_list[j,i]      = total_work
ergo_average_6 = np.average(ergo_list,axis=0)
iteration_average_6 = np.average(iteration_list,axis=0)
work_average_6 = np.average(work_list,axis=0)
standard_deviation_ergo_6 = np.std(ergo_list,axis=0,ddof=1)

m = 7           
depth_passs = [2]
dir_name_m = dir_name + f"m{m}/"
ergo_list = np.zeros((len(passive_seeds),reps+1))
iteration_list = np.zeros((len(passive_seeds),reps+1))
work_list = np.zeros((len(passive_seeds),reps+1))
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name_m + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            total_work = quantities[1]
            ergotropy = quantities[3]
            iteration = quantities[4]
            ergo_list[j,i] = ergotropy
            iteration_list[j,i] = iteration
            work_list[j,i]      = total_work
ergo_average_7 = np.average(ergo_list,axis=0)
iteration_average_7 = np.average(iteration_list,axis=0)
work_average_7 = np.average(work_list,axis=0)
standard_deviation_ergo_7 = np.std(ergo_list,axis=0,ddof=1)

####################################################### Begin plotting #######################################################
plt.rc('font', family='serif')#, serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=60)
plt.rc('ytick', labelsize=60)
plt.rc('axes', labelsize=60)
plt.rc('legend', fontsize=60)
plt.rc('legend', handlelength=2)
plt.rc('font', size=60)
linestyles = ["-", "--", "dotted"]
colors = ['tab:orange','tab:blue', 'tab:red', 'tab:brown','tab:purple', 'tab:green', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
markers = ["o", "v", "s", "d", "*"]
##### Specify columns and rows
fig = plt.figure(figsize=(27,13))
gs = fig.add_gridspec(nrows=1,ncols=2) #height_ratios=[1,1], width_ratios=[1,1])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

M_list = [1,2,3,4,5,6,7]


iteration_average_list = [iteration_average_1[2],iteration_average_2[2],iteration_average_3[2],iteration_average_4[2],iteration_average_5[2],iteration_average_6[2],iteration_average_7[2]]
ax1.plot(M_list,iteration_average_list, ms=15,marker = "o",ls='-',lw=5,color = "blue")

standard_deviation_ergo_list= [standard_deviation_ergo_1[2],standard_deviation_ergo_2[2],standard_deviation_ergo_3[2],standard_deviation_ergo_4[2],standard_deviation_ergo_5[2],standard_deviation_ergo_6[2],standard_deviation_ergo_7[2]]
ax2.plot(M_list,standard_deviation_ergo_list, ms=15,marker = "o",ls='-',lw=5,color = "blue")



ax1.text(0.05, 0.93,'a)',transform=ax1.transAxes,size=60, weight='bold')
ax2.text(0.05, 0.93,'b)',transform=ax2.transAxes,size=60, weight='bold')

ax1.set_xlabel(r"$M$")
ax1.set_ylabel(r"Iteration")
ax1.set_xticks([1,2,3,4,5,6,7])
ax2.set_xlabel(r"$M$")
ax2.set_ylabel(r"$\sigma(\mathcal{E})$")
ax2.set_yscale('log')
ax2.set_xticks([1,2,3,4,5,6,7])
ax1.tick_params('both', length=10, width=2, which='major')
ax2.tick_params('both', length=10, width=2, which='major')
# fig.suptitle(r"Keeping the magnetic field, $N=%s,h=%s,J=%s$, without noise" %(n,-h,-J))

##### Close and save figure
plt.tight_layout()
plt.savefig(dir_name+f"pvqd_statevector_Mscale.eps", dpi=300)
plt.show()
#plt.close()

