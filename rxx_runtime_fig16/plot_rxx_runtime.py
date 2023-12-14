import numpy as np 
import pickle
import matplotlib.pyplot as plt
import itertools
####################################################### Parameters of the systems #######################################################
n = 2           # number of qubits (total cells)
m = 1           # number of cells we want to extract the energy (M <= N)
h  = -0.6       # a transverse magnetic field
J  = -2.0       # J is nearest-neighbor interaction strength
reps = 16
max_time = 1.6 
times = np.linspace(0.0,max_time,reps+1)

####################################################### Define variable and directory name #######################################################
dir_name = f"results_noisy_simulation/n{n}_J{J}_h{h}/time{max_time}_reps{reps}/m{m}/"
index_reps = np.arange(reps+1)
passive_seeds = np.arange(1,101)
ergo_list = np.zeros((len(passive_seeds),reps+1))
work_list = np.zeros((len(passive_seeds),reps+1))

dir_name_ibm = f"results/n{n}_J{J}_h{h}/time{max_time}_reps{reps}/m{m}/"
times_ibm = [0.6,1.0]
index_ibm = [6,10]
ergo_list_ibm = np.zeros((len(times_ibm)))
work_list_ibm = np.zeros((len(times_ibm)))
####################################################### Load data of matlab #######################################################

####################################################### Load data qiskit #######################################################
for i,index_rep in enumerate(index_reps):
    for j,passive_seed in enumerate(passive_seeds):
        dir_name_time = dir_name + f"time_index{index_rep}/"
        ######## Load data of quantities
        dir_name_quantities = dir_name_time + "quantities/"
        save_name_quantities = dir_name_quantities + f"quantities_seed{passive_seed}"
        quantities = np.load(save_name_quantities+'.npy')
        ergotropy = quantities[3]
        total_work = quantities[1]
        ergo_list[j,i] = ergotropy
        work_list[j,i] = total_work

for i,index_rep in enumerate(index_ibm):
    dir_name_time_ibm = dir_name_ibm + f"time_index{index_rep}/"
    ######## Load data of quantities
    dir_name_quantities_ibm = dir_name_time_ibm + "quantities/"
    save_name_quantities_ibm = dir_name_quantities_ibm + f"quantities_seed{97}"
    quantities = np.load(save_name_quantities_ibm+'.npy')
    ergotropy = quantities[3]
    total_work = quantities[1]
    ergo_list_ibm[i] = ergotropy
    work_list_ibm[i] = total_work

matlab_ergo = []
matlab_time = []
matlab_work = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}_J2_step200.txt",'r')
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
plt.rc('legend', fontsize=24)
plt.rc('legend', handlelength=2)
plt.rc('font', size=25)

linestyles = ["-", "--", "dotted"]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
markers = ["o", "v", "s", "d", "*","x"]
colors = ['#005c66', '#709a32', '#ffc35a', '#845e25', '#e9002c']

fig = plt.figure(figsize=(12,10))
gs = fig.add_gridspec(nrows=1,ncols=1) #height_ratios=[1,1], width_ratios=[1,1])
ax1 = fig.add_subplot(gs[0,0])

ergo_average = np.average(ergo_list,axis=0)
standard_deviation_ergo = np.std(ergo_list,axis=0,ddof=1)
work_average = np.average(work_list,axis=0)
standard_deviation_work = np.std(work_list,axis=0,ddof=1)

ax1.plot(matlab_time, matlab_work, ls=linestyles[1],lw=3,color = 'dimgrey',label=r"$W$-ED",zorder=1)
ax1.errorbar(times, work_average, yerr=standard_deviation_ergo,capsize =6 ,ms=8,marker = 'v',ls='',label=r"$W$-FakePerth",color='limegreen',zorder=2)
ax1.plot(times_ibm,work_list_ibm,ms=8, marker = 'D',ls='',label=r"$W$-IBMPerth",color=colors[4],zorder=3)
ax1.text(0.05, 0.9,r"$N=%s,M=%s$" %(n,m),transform=ax1.transAxes,size=30, weight='bold')


ax1.plot(matlab_time, matlab_ergo, ls=linestyles[0],lw=3,label=r"$\mathcal{E}$-ED", color='orange',zorder=1)
ax1.errorbar(times, ergo_average, yerr=standard_deviation_ergo,capsize =6 ,ms=7, marker = 'o',ls='',label=r"$\mathcal{E}$-FakePerth",color='dodgerblue',zorder=2)
ax1.plot(times_ibm,ergo_list_ibm,ms=10, marker = 'h',ls='',label=r"$\mathcal{E}$-IBMPerth",color=colors[3],zorder=3)
# fig.suptitle(r"Turning off the magnetic field, $N=%s,M=%s,h=%s,J=%s$" %(n,m,-h,-J))
# plt.yscale('log')

##### Set label of x,y axis, label of plots and super title of figure

ax1.set_xlabel(r"$t$")
ax1.set_ylabel("Energy")
ax1.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])

##### Close and save figure
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,2,1,3,4,5]
ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc=1) 
ax1.tick_params('both', length=10, width=2, which='major')
plt.tight_layout()
plt.savefig(dir_name_ibm+f"rxx_runtime_N{n}_M{m}_ver2.eps", dpi=300)
plt.show()
#plt.close()




