import numpy as np 
import pickle
import matplotlib.pyplot as plt
import itertools
####################################################### Parameters of the systems #######################################################
n = 6            # number of qubits (total cells)
h  = -0.6        # a transverse magnetic field
J  = -2.0        # J is nearest-neighbor interaction strength
reps = 16        # number of time-step for pvqd method
max_time = 1.6      # evolution time
times = np.linspace(0.0,max_time,reps+1) 
shots = 2048
dir_name = "results/"
dir_name = dir_name + f"n{n}_J{J}_h{h}/time{max_time}_reps{reps}_shots{shots}/"
index_reps = np.arange(reps+1)
seeds = np.arange(1,101)
# seeds = np.delete(seeds,3)


m = 1
dir_name_m = dir_name + f"m{m}/"
work_list = np.zeros((len(seeds),reps+1))
ergo_list = np.zeros((len(seeds),reps+1))
for i,index_rep in enumerate(index_reps):
    for j,seed in enumerate(seeds):
        dir_name_time = dir_name_m + f"time_index{index_rep}/"
        dir_name_quantities = dir_name_time + "quantities/"
        save_name_quantities = dir_name_quantities + f"quantities_seed{seed}"
        quantities = np.load(save_name_quantities+'.npy',allow_pickle=True)
        mean_energy = quantities[0]
        total_work = quantities[1]
        ergotropy = quantities[3]
        work_list[j,i] = total_work
        ergo_list[j,i] = ergotropy
matlab_ergo_1 = []
matlab_time = []
matlab_work_1 = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}_J{int(-J)}_step200.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time.append(float(row[0]))
    matlab_ergo_1.append(float(row[2]))
    matlab_work_1.append(float(row[1]))
average_ergo_1 = np.average(ergo_list,axis=0)
standard_deviation_ergo_1 = np.std(ergo_list,axis=0,ddof=1)

m = 2
dir_name_m = dir_name + f"m{m}/"
work_list = np.zeros((len(seeds),reps+1))
ergo_list = np.zeros((len(seeds),reps+1))
for i,index_rep in enumerate(index_reps):
    for j,seed in enumerate(seeds):
        dir_name_time = dir_name_m + f"time_index{index_rep}/"
        dir_name_quantities = dir_name_time + "quantities/"
        save_name_quantities = dir_name_quantities + f"quantities_seed{seed}"
        quantities = np.load(save_name_quantities+'.npy',allow_pickle=True)
        mean_energy = quantities[0]
        total_work = quantities[1]
        ergotropy = quantities[3]
        work_list[j,i] = total_work
        ergo_list[j,i] = ergotropy
matlab_ergo_2 = []
matlab_work_2 = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}_J{int(-J)}_step200.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_ergo_2.append(float(row[2]))
    matlab_work_2.append(float(row[1]))
average_ergo_2 = np.average(ergo_list,axis=0)
standard_deviation_ergo_2 = np.std(ergo_list,axis=0,ddof=1)

m = 3
dir_name_m = dir_name + f"m{m}/"
work_list = np.zeros((len(seeds),reps+1))
ergo_list = np.zeros((len(seeds),reps+1))
for i,index_rep in enumerate(index_reps):
    for j,seed in enumerate(seeds):
        dir_name_time = dir_name_m + f"time_index{index_rep}/"
        dir_name_quantities = dir_name_time + "quantities/"
        save_name_quantities = dir_name_quantities + f"quantities_seed{seed}"
        quantities = np.load(save_name_quantities+'.npy',allow_pickle=True)
        mean_energy = quantities[0]
        total_work = quantities[1]
        ergotropy = quantities[3]
        work_list[j,i] = total_work
        ergo_list[j,i] = ergotropy
matlab_ergo_3 = []
matlab_work_3 = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}_J{int(-J)}_step200.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_ergo_3.append(float(row[2]))
    matlab_work_3.append(float(row[1]))
average_ergo_3 = np.average(ergo_list,axis=0)
standard_deviation_ergo_3 = np.std(ergo_list,axis=0,ddof=1)

m = 4
dir_name_m = dir_name + f"m{m}/"
work_list = np.zeros((len(seeds),reps+1))
ergo_list = np.zeros((len(seeds),reps+1))
for i,index_rep in enumerate(index_reps):
    for j,seed in enumerate(seeds):
        dir_name_time = dir_name_m + f"time_index{index_rep}/"
        dir_name_quantities = dir_name_time + "quantities/"
        save_name_quantities = dir_name_quantities + f"quantities_seed{seed}"
        quantities = np.load(save_name_quantities+'.npy',allow_pickle=True)
        mean_energy = quantities[0]
        total_work = quantities[1]
        ergotropy = quantities[3]
        work_list[j,i] = total_work
        ergo_list[j,i] = ergotropy
matlab_ergo_4 = []
matlab_work_4 = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}_J{int(-J)}_step200.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_ergo_4.append(float(row[2]))
    matlab_work_4.append(float(row[1]))
average_ergo_4 = np.average(ergo_list,axis=0)
standard_deviation_ergo_4 = np.std(ergo_list,axis=0,ddof=1)

m = 5
dir_name_m = dir_name + f"m{m}/"
work_list = np.zeros((len(seeds),reps+1))
ergo_list = np.zeros((len(seeds),reps+1))
for i,index_rep in enumerate(index_reps):
    for j,seed in enumerate(seeds):
        dir_name_time = dir_name_m + f"time_index{index_rep}/"
        dir_name_quantities = dir_name_time + "quantities/"
        save_name_quantities = dir_name_quantities + f"quantities_seed{seed}"
        quantities = np.load(save_name_quantities+'.npy',allow_pickle=True)
        mean_energy = quantities[0]
        total_work = quantities[1]
        ergotropy = quantities[3]
        work_list[j,i] = total_work
        ergo_list[j,i] = ergotropy
matlab_ergo_5 = []
matlab_work_5 = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}_J{int(-J)}_step200.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_ergo_5.append(float(row[2]))
    matlab_work_5.append(float(row[1]))
average_ergo_5 = np.average(ergo_list,axis=0)
standard_deviation_ergo_5 = np.std(ergo_list,axis=0,ddof=1)




####################################################### Begin plotting #######################################################
plt.rc('font', family='serif')#, serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
plt.rc('axes', labelsize=30)
plt.rc('legend', fontsize=20)
plt.rc('legend', handlelength=2)
plt.rc('font', size=25)
linestyles = ["-", "--", "dotted"]
colors = ['tab:orange','tab:blue', 'tab:red', 'tab:brown','tab:purple', 'tab:green', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
markers = ["o", "v", "s", "d", "*"]
##### Specify columns and rows
fig = plt.figure(figsize=(20,10))
gs = fig.add_gridspec(nrows=2,ncols=6) #height_ratios=[1,1], width_ratios=[1,1])
ax1 = fig.add_subplot(gs[0,0:2])
ax2 = fig.add_subplot(gs[0,2:4])
ax3 = fig.add_subplot(gs[0,4:])
ax4 = fig.add_subplot(gs[1,1:3])
ax5 = fig.add_subplot(gs[1,3:5])

ax1.plot(matlab_time, matlab_work_1, ls=linestyles[1],lw=2,label="Work",color = 'dimgrey')
ax2.plot(matlab_time, matlab_work_2, ls=linestyles[1],lw=2,label="Work",color = 'dimgrey')
ax3.plot(matlab_time, matlab_work_3, ls=linestyles[1],lw=2,label="Work",color = 'dimgrey')
ax4.plot(matlab_time, matlab_work_4, ls=linestyles[1],lw=2,label="Work",color = 'dimgrey')
ax5.plot(matlab_time, matlab_work_5, ls=linestyles[1],lw=2,label="Work",color = 'dimgrey')

ax1.plot(matlab_time, matlab_ergo_1, ls=linestyles[0],lw=3,label="ED",color = colors[0])
ax2.plot(matlab_time, matlab_ergo_2, ls=linestyles[0],lw=3,label="ED",color = colors[0])
ax3.plot(matlab_time, matlab_ergo_3, ls=linestyles[0],lw=3,label="ED",color = colors[0])
ax4.plot(matlab_time, matlab_ergo_4, ls=linestyles[0],lw=3,label="ED",color = colors[0])
ax5.plot(matlab_time, matlab_ergo_5, ls=linestyles[0],lw=3,label="ED",color = colors[0])


ax1.errorbar(times, average_ergo_1, yerr=standard_deviation_ergo_1,capsize =6 ,ms=6, marker = markers[0],ls='',color = colors[1])
ax2.errorbar(times, average_ergo_2, yerr=standard_deviation_ergo_2,capsize =6 ,ms=6, marker = markers[0],ls='',color = colors[1],label="FakePerth")
ax3.errorbar(times, average_ergo_3, yerr=standard_deviation_ergo_3,capsize =6 ,ms=6, marker = markers[0],ls='',color = colors[1])
ax4.errorbar(times, average_ergo_4, yerr=standard_deviation_ergo_4,capsize =6 ,ms=6, marker = markers[0],ls='',color = colors[1])
ax5.errorbar(times, average_ergo_5, yerr=standard_deviation_ergo_5,capsize =6 ,ms=6, marker = markers[0],ls='',color = colors[1])


ax1.text(0.05, 0.9,'a)',transform=ax1.transAxes,size=30, weight='bold')
ax2.text(0.05, 0.9,'b)',transform=ax2.transAxes,size=30, weight='bold')
ax3.text(0.05, 0.9,'c)',transform=ax3.transAxes,size=30, weight='bold')
ax4.text(0.05, 0.9,'d)',transform=ax4.transAxes,size=30, weight='bold')
ax5.text(0.05, 0.9,'e)',transform=ax5.transAxes,size=30, weight='bold')

ax1.text(0.77, 0.9,r'$M=1$',transform=ax1.transAxes,size=30, weight='bold')
ax2.text(0.77, 0.9,r'$M=2$',transform=ax2.transAxes,size=30, weight='bold')
ax3.text(0.77, 0.9,r'$M=3$',transform=ax3.transAxes,size=30, weight='bold')
ax4.text(0.77, 0.9,r'$M=4$',transform=ax4.transAxes,size=30, weight='bold')
ax5.text(0.77, 0.9,r'$M=5$',transform=ax5.transAxes,size=30, weight='bold')

ax1.set_xlabel(r"$t$")
ax1.set_ylabel(r"$\mathcal{E}$")
ax1.set_xticks([0.0,0.5,1.0,1.5])

ax2.set_xlabel(r"$t$")
ax2.set_ylabel(r"$\mathcal{E}$")
ax2.set_xticks([0.0,0.5,1.0,1.5])

ax3.set_xlabel(r"$t$")
ax3.set_ylabel(r"$\mathcal{E}$")
ax3.set_xticks([0.0,0.5,1.0,1.5])

ax4.set_xlabel(r"$t$")
ax4.set_ylabel(r"$\mathcal{E}$")
ax4.set_xticks([0.0,0.5,1.0,1.5])

ax5.set_xlabel(r"$t$")
ax5.set_ylabel(r"$\mathcal{E}$")
ax5.set_xticks([0.0,0.5,1.0,1.5])

# fig.suptitle(r"Turning off the magnetic field, $N=%s,h=%s,J=%s$, noisy simulation" %(n,-h,-J))

##### Close and save figure
ax2.legend(loc=8)
# ax2.legend()
# ax3.legend()
# ax4.legend()
plt.tight_layout()
plt.savefig(dir_name+f"rxx_noisy_depth1.eps", dpi=300)
plt.show()
#plt.close()