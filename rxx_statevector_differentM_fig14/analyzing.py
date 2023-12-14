import numpy as np 
import pickle
import matplotlib.pyplot as plt
import itertools
####################################################### Parameters of the systems #######################################################
n = 10           # number of qubits (total cells)
m = 9           # number of cells we want to extract the energy (M <= N)
h  = -0.6       # a transverse magnetic field
J  = -2.0       # J is nearest-neighbor interaction strength
reps = 16
max_time = 1.6 
times = np.linspace(0.0,max_time,reps+1)

####################################################### Define variable and directory name #######################################################
dir_name = "results/"
dir_name = dir_name + f"n{n}_J{J}_h{h}/"
dir_name = dir_name + f"m{m}/"

index_reps = np.arange(reps+1)
passive_seeds = np.arange(1,101)
depth_passs = [1,2,3]
ergo_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
# error_ergo = np.zeros((reps+1,len(depth_passs)))
####################################################### Load data of matlab #######################################################

####################################################### Load data qiskit #######################################################
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            ######## Load data of quantities
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            ergotropy = quantities[3]
            ergo_list[j,i,k] = ergotropy

            ######## Relative/Absolute error calculation 
            # error_ergo[j,i] = abs((ergotropy - ergotropy_matlab[i])/ergotropy_matlab[i])
            # error_ergo[j,i,k] = abs(ergotropy - ergotropy_matlab[i])

matlab_ergo = []
matlab_time = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}_J2_step200.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time.append(float(row[0]))
    matlab_ergo.append(float(row[2]))
####################################################### Begin plotting #######################################################
##### Define font, size, type of text, ls, color, markerstyle, etc
plt.rc('font', family='serif')#, serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=16)
plt.rc('legend', fontsize=14)
plt.rc('legend', handlelength=2)
plt.rc('font', size=18)
#plt.figure(figsize=(10,6))
linestyles = ["-", "--", "dotted"]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
markers = ["o", "v", "s", "d", "*","x"]
plt.plot(matlab_time, matlab_ergo, ls=linestyles[0],lw=3,label="Exact results")
ergo_average = np.average(ergo_list,axis=0)
standard_deviation_ergo = np.std(ergo_list,axis=0,ddof=1)

for k,depth_pass in enumerate(depth_passs):
    # plt.plot(times, ergo_average[:,k] ,ms=8, marker = markers[k],color = colors[k+1],ls='')
    plt.errorbar(times, ergo_average[:,k], yerr=standard_deviation_ergo[:,k],capsize =4 ,ms=4, marker = markers[k],ls='',label=f"depth = {depth_pass}")
# plt.yscale('log')

##### Set label of x,y axis, label of plots and super title of figure
plt.xlabel(r"time")
plt.ylabel(r"$\mathcal{E}$")

##### Close and save figure
plt.legend(bbox_to_anchor=(1, 0.5))
plt.subplots_adjust(left = 0.144, right=0.641, top = 0.914, bottom = 0.132)
plt.title(f'M = {m}')
plt.savefig(dir_name+f"N{n}_M{m}.png", dpi=300)
plt.show()
#plt.close()




