import numpy as np 
import pickle
import matplotlib.pyplot as plt
import itertools
####################################################### Parameters of the systems #######################################################
n = 4           # number of qubits (total cells)
m = 2           # number of cells we want to extract the energy (M <= N)
h  = -0.6       # a transverse magnetic field
J  = -2.0       # J is nearest-neighbor interaction strength

dir_name_ibm = f"data_runtime/results/n{n}_J{J}_h{h}/m{m}/"
times_ibm = [0.2,0.4,0.6,0.7,0.9,1.2]
index_ibm = [2,4,6,7,9,12]
ergo_list = np.zeros((len(times_ibm),551))
energy    = np.zeros((len(times_ibm),275))

####################################################### Load data of matlab #######################################################
##### N4_M2
total_work_matlab = [0,0.137692802859178,0.481642020280141,0.868092948670199,1.13951814150673,1.23411507493769,1.20564922064944,1.16593797096761,1.19581788123833,1.28874849529406,1.36318930549216,1.32850425836185,1.15289792997237,0.885628473450649,0.623213843819488]
ergotropy_matlab = [0,0.0905509633358443,0.302824311369582,0.498773733400337,0.554624753583859,0.823958963661795,0.958867488622713,1.01757239788005,1.07137375393794,1.11980350692653,1.09832469495944,0.935044103367898,0.611243602813555,0.289770264732029,0.121038914867986]

####################################################### Load data qiskit #######################################################
for i,index_rep in enumerate(index_ibm):
    dir_name_time = dir_name_ibm + f"time_index{index_rep}/"
    ######## Load data of quantities
    dir_name_quantities = dir_name_time + "quantities/"
    save_name_quantities = dir_name_quantities+ f"quantities_seed{34}"
    quantities = np.load(save_name_quantities+'.npy')
    mean_energy = quantities[0]
    total_work = quantities[1]
    # ergotropy = quantities[3]
    # work_list[j,i] = total_work
    iteration = int(quantities[4])

    dir_name_inter_info = dir_name_time + "intermediate_info/"
    save_name_intermediate_info = dir_name_inter_info + f"intermediate_info_seed{34}"
    with open(save_name_intermediate_info+'.pkl', 'rb') as handle:
        intermediate_info = pickle.load(handle)
        passive_energy = intermediate_info['energy']
    for k in range(iteration):
        ergotropy = mean_energy - passive_energy[k]
        ergo_list[i,k] = ergotropy
    for j in range(1,551,2):
        x = int((j-1)/2)
        # print(x)
        energy[i,x] = 0.5*(ergo_list[i,j] + ergo_list[i,j+1])
        

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
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
# markers = ["o", "v", "s", "d", "*"]
marker = itertools.cycle(("o", "v", "s", "d", "*"))

##### Specify columns and rows
fig = plt.figure(figsize=(12,14))
gs = fig.add_gridspec(nrows=2,ncols=1) #height_ratios=[1,1], width_ratios=[1,1])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0])

ax1.axhline(y=ergotropy_matlab[2],color = "black" ,ls=linestyles[1],lw=3,label="ED")
ax2.axhline(y=ergotropy_matlab[4], color="black",ls=linestyles[1],lw=3,label="ED")
ax1.plot(energy[0,:], ms=6, marker = "o",ls='-',color = 'blue')
ax2.plot(energy[1,:], ms=6, marker = "o",ls='-',color = 'blue')

ax1.text(0.75, 0.1,f'a) t = 0.2',transform=ax1.transAxes,size=35, weight='bold')
ax2.text(0.75, 0.1,f'b) t = 0.4',transform=ax2.transAxes,size=35, weight='bold')

##### Set label of x,y axis, label of plots and super title of figure
ax1.set_ylabel(r"$\mathcal{E}$")


ax2.set_xlabel(r"Iteration")
ax2.set_ylabel(r"$\mathcal{E}$")



##### Close and save figure
ax1.legend(loc=8)
ax1.tick_params('both', length=10, width=2, which='major')
ax2.tick_params('both', length=10, width=2, which='major')
plt.tight_layout()
plt.savefig(dir_name_ibm+f"runtime_iteration.eps", dpi=300)
plt.show()
#plt.close()




