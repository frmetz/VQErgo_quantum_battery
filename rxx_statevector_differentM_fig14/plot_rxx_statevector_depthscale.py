import numpy as np 
import pickle
import matplotlib.pyplot as plt
import itertools
####################################################### Loading data #######################################################
n = 10           # number of qubits (total cells)
h  = -0.6       # a transverse magnetic field
J  = -2.0       # J is nearest-neighbor interaction strength
reps = 16
max_time = 1.6 
times = np.linspace(0.0,max_time,reps+1)
dir_name = "results/"
dir_name = dir_name + f"n{n}_J{J}_h{h}/"
index_reps = np.arange(reps+1)
passive_seeds = np.arange(1,101)

m = 1    
dir_name_m = dir_name + f"m{m}/"
depth_passs = [1,2]
ergo_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name_m + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            ######## Load data of quantities
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            ergotropy = quantities[3]
            ergo_list[j,i,k] = ergotropy
matlab_ergo_1 = []
matlab_time_1 = []
matlab_work_1 = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}_J2_step200.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time_1.append(float(row[0]))
    matlab_ergo_1.append(float(row[2]))
    matlab_work_1.append(float(row[1]))
ergo_average_1 = np.average(ergo_list,axis=0)
standard_deviation_ergo_1 = np.std(ergo_list,axis=0,ddof=1)

m = 2    
dir_name_m = dir_name + f"m{m}/"
depth_passs = [1,2]
ergo_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name_m + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            ######## Load data of quantities
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            ergotropy = quantities[3]
            ergo_list[j,i,k] = ergotropy
matlab_ergo_2 = []
matlab_time_2 = []
matlab_work_2 = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}_J2_step200.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time_2.append(float(row[0]))
    matlab_ergo_2.append(float(row[2]))
    matlab_work_2.append(float(row[1]))
ergo_average_2 = np.average(ergo_list,axis=0)
standard_deviation_ergo_2 = np.std(ergo_list,axis=0,ddof=1)


m = 3    
dir_name_m = dir_name + f"m{m}/"
depth_passs = [1,2]
ergo_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name_m + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            ######## Load data of quantities
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            ergotropy = quantities[3]
            ergo_list[j,i,k] = ergotropy
matlab_ergo_3 = []
matlab_time_3 = []
matlab_work_3 = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}_J2_step200.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time_3.append(float(row[0]))
    matlab_ergo_3.append(float(row[2]))
    matlab_work_3.append(float(row[1]))
ergo_average_3 = np.average(ergo_list,axis=0)
standard_deviation_ergo_3 = np.std(ergo_list,axis=0,ddof=1)


m = 4    
dir_name_m = dir_name + f"m{m}/"
depth_passs = [1,2]
ergo_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name_m + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            ######## Load data of quantities
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            ergotropy = quantities[3]
            ergo_list[j,i,k] = ergotropy
matlab_ergo_4 = []
matlab_time_4 = []
matlab_work_4 = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}_J2_step200.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time_4.append(float(row[0]))
    matlab_ergo_4.append(float(row[2]))
    matlab_work_4.append(float(row[1]))
ergo_average_4 = np.average(ergo_list,axis=0)
standard_deviation_ergo_4 = np.std(ergo_list,axis=0,ddof=1)


m = 5    
dir_name_m = dir_name + f"m{m}/"
depth_passs = [1,2,3]
ergo_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name_m + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            ######## Load data of quantities
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            ergotropy = quantities[3]
            ergo_list[j,i,k] = ergotropy
matlab_ergo_5 = []
matlab_time_5 = []
matlab_work_5 = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}_J2_step200.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time_5.append(float(row[0]))
    matlab_ergo_5.append(float(row[2]))
    matlab_work_5.append(float(row[1]))
ergo_average_5 = np.average(ergo_list,axis=0)
standard_deviation_ergo_5 = np.std(ergo_list,axis=0,ddof=1)


m = 6    
dir_name_m = dir_name + f"m{m}/"
depth_passs = [1,2,3]
ergo_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name_m + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            ######## Load data of quantities
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            ergotropy = quantities[3]
            ergo_list[j,i,k] = ergotropy
matlab_ergo_6 = []
matlab_time_6 = []
matlab_work_6 = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}_J2_step200.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time_6.append(float(row[0]))
    matlab_ergo_6.append(float(row[2]))
    matlab_work_6.append(float(row[1]))
ergo_average_6 = np.average(ergo_list,axis=0)
standard_deviation_ergo_6 = np.std(ergo_list,axis=0,ddof=1)


m = 7    
dir_name_m = dir_name + f"m{m}/"
depth_passs = [1,2,3]
ergo_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name_m + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            ######## Load data of quantities
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            ergotropy = quantities[3]
            ergo_list[j,i,k] = ergotropy
matlab_ergo_7 = []
matlab_time_7 = []
matlab_work_7 = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}_J2_step200.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time_7.append(float(row[0]))
    matlab_ergo_7.append(float(row[2]))
    matlab_work_7.append(float(row[1]))
ergo_average_7 = np.average(ergo_list,axis=0)
standard_deviation_ergo_7 = np.std(ergo_list,axis=0,ddof=1)


m = 8    
dir_name_m = dir_name + f"m{m}/"
depth_passs = [1,2,3]
ergo_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name_m + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            ######## Load data of quantities
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            ergotropy = quantities[3]
            ergo_list[j,i,k] = ergotropy
matlab_ergo_8 = []
matlab_time_8 = []
matlab_work_8 = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}_J2_step200.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time_8.append(float(row[0]))
    matlab_ergo_8.append(float(row[2]))
    matlab_work_8.append(float(row[1]))
ergo_average_8 = np.average(ergo_list,axis=0)
standard_deviation_ergo_8 = np.std(ergo_list,axis=0,ddof=1)


m = 9    
dir_name_m = dir_name + f"m{m}/"
depth_passs = [1,2,3]
ergo_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name_m + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            ######## Load data of quantities
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            ergotropy = quantities[3]
            ergo_list[j,i,k] = ergotropy
matlab_ergo_9 = []
matlab_time_9 = []
matlab_work_9 = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}_J2_step200.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time_9.append(float(row[0]))
    matlab_ergo_9.append(float(row[2]))
    matlab_work_9.append(float(row[1]))
ergo_average_9 = np.average(ergo_list,axis=0)
standard_deviation_ergo_9 = np.std(ergo_list,axis=0,ddof=1)


################################################### Begin plotting #########################################
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
fig = plt.figure(figsize=(20,15))
gs = fig.add_gridspec(nrows=3,ncols=3) #height_ratios=[1,1], width_ratios=[1,1])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[1,0])
ax5 = fig.add_subplot(gs[1,1])
ax6 = fig.add_subplot(gs[1,2])
ax7 = fig.add_subplot(gs[2,0])
ax8 = fig.add_subplot(gs[2,1])
ax9 = fig.add_subplot(gs[2,2])

ax1.plot(matlab_time_1, matlab_work_1, ls=linestyles[1],lw=2,label="Work",color = 'dimgrey')
ax2.plot(matlab_time_2, matlab_work_2, ls=linestyles[1],lw=2,label="Work",color = 'dimgrey')
ax3.plot(matlab_time_3, matlab_work_3, ls=linestyles[1],lw=2,label="Work",color = 'dimgrey')
ax4.plot(matlab_time_4, matlab_work_4, ls=linestyles[1],lw=2,label="Work",color = 'dimgrey')
ax5.plot(matlab_time_5, matlab_work_5, ls=linestyles[1],lw=2,label="Work",color = 'dimgrey')
ax6.plot(matlab_time_6, matlab_work_6, ls=linestyles[1],lw=2,label="Work",color = 'dimgrey')
ax7.plot(matlab_time_7, matlab_work_7, ls=linestyles[1],lw=2,color = 'dimgrey')
ax8.plot(matlab_time_8, matlab_work_8, ls=linestyles[1],lw=2,color = 'dimgrey')
ax9.plot(matlab_time_9, matlab_work_9, ls=linestyles[1],lw=2,color = 'dimgrey')

ax1.plot(matlab_time_1, matlab_ergo_1, ls=linestyles[0],lw=3,label="ED",color = colors[0])
ax2.plot(matlab_time_2, matlab_ergo_2, ls=linestyles[0],lw=3,label="ED",color = colors[0])
ax3.plot(matlab_time_3, matlab_ergo_3, ls=linestyles[0],lw=3,label="ED",color = colors[0])
ax4.plot(matlab_time_4, matlab_ergo_4, ls=linestyles[0],lw=3,label="ED",color = colors[0])
ax5.plot(matlab_time_5, matlab_ergo_5, ls=linestyles[0],lw=3,label="ED",color = colors[0])
ax6.plot(matlab_time_6, matlab_ergo_6, ls=linestyles[0],lw=3,label="ED",color = colors[0])
ax7.plot(matlab_time_7, matlab_ergo_7, ls=linestyles[0],lw=3,label="ED",color = colors[0])
ax8.plot(matlab_time_8, matlab_ergo_8, ls=linestyles[0],lw=3,label="ED",color = colors[0])
ax9.plot(matlab_time_9, matlab_ergo_9, ls=linestyles[0],lw=3,label="ED",color = colors[0])
depth_pass_1 =[1,2]
depth_pass_2 =[1,2,3]
for k,depth_pass in enumerate(depth_pass_1):
    ax1.errorbar(times, ergo_average_1[:,k], yerr=standard_deviation_ergo_1[:,k],capsize =6 ,ms=6, marker = markers[k],ls='',label=f"depth = {depth_pass}",color = colors[k+1])
    ax2.errorbar(times, ergo_average_2[:,k], yerr=standard_deviation_ergo_2[:,k],capsize =6 ,ms=6, marker = markers[k],ls='',label=f"depth = {depth_pass}",color = colors[k+1])
    ax3.errorbar(times, ergo_average_3[:,k], yerr=standard_deviation_ergo_3[:,k],capsize =6 ,ms=6, marker = markers[k],ls='',label=f"depth = {depth_pass}",color = colors[k+1])
    ax4.errorbar(times, ergo_average_4[:,k], yerr=standard_deviation_ergo_4[:,k],capsize =6 ,ms=6, marker = markers[k],ls='',label=f"depth = {depth_pass}",color = colors[k+1])

for k,depth_pass in enumerate(depth_pass_2):
    ax5.errorbar(times, ergo_average_5[:,k], yerr=standard_deviation_ergo_5[:,k],capsize =6 ,ms=6, marker = markers[k],ls='',label=f"depth = {depth_pass}",color = colors[k+1])
    ax6.errorbar(times, ergo_average_6[:,k], yerr=standard_deviation_ergo_6[:,k],capsize =6 ,ms=6, marker = markers[k],ls='',label=f"depth = {depth_pass}",color = colors[k+1])
    ax7.errorbar(times, ergo_average_7[:,k], yerr=standard_deviation_ergo_7[:,k],capsize =6 ,ms=6, marker = markers[k],ls='',label=f"depth = {depth_pass}",color = colors[k+1])
    ax8.errorbar(times, ergo_average_8[:,k], yerr=standard_deviation_ergo_8[:,k],capsize =6 ,ms=6, marker = markers[k],ls='',label=f"depth = {depth_pass}",color = colors[k+1])
    ax9.errorbar(times, ergo_average_9[:,k], yerr=standard_deviation_ergo_9[:,k],capsize =6 ,ms=6, marker = markers[k],ls='',label=f"depth = {depth_pass}",color = colors[k+1])



ax1.text(0.05, 0.9,'a)',transform=ax1.transAxes,size=30, weight='bold')
ax2.text(0.05, 0.9,'b)',transform=ax2.transAxes,size=30, weight='bold')
ax3.text(0.05, 0.9,'c)',transform=ax3.transAxes,size=30, weight='bold')
ax4.text(0.05, 0.9,'d)',transform=ax4.transAxes,size=30, weight='bold')
ax5.text(0.05, 0.9,'e)',transform=ax5.transAxes,size=30, weight='bold')
ax6.text(0.05, 0.9,'f)',transform=ax6.transAxes,size=30, weight='bold')
ax7.text(0.05, 0.9,'g)',transform=ax7.transAxes,size=30, weight='bold')
ax8.text(0.05, 0.9,'h)',transform=ax8.transAxes,size=30, weight='bold')
ax9.text(0.05, 0.9,'i)',transform=ax9.transAxes,size=30, weight='bold')

ax1.text(0.76, 0.9,r'$M=1$',transform=ax1.transAxes,size=30, weight='bold')
ax2.text(0.76, 0.9,r'$M=2$',transform=ax2.transAxes,size=30, weight='bold')
ax3.text(0.76, 0.9,r'$M=3$',transform=ax3.transAxes,size=30, weight='bold')
ax4.text(0.76, 0.9,r'$M=4$',transform=ax4.transAxes,size=30, weight='bold')
ax5.text(0.76, 0.9,r'$M=5$',transform=ax5.transAxes,size=30, weight='bold')
ax6.text(0.76, 0.9,r'$M=6$',transform=ax6.transAxes,size=30, weight='bold')
ax7.text(0.76, 0.9,r'$M=7$',transform=ax7.transAxes,size=30, weight='bold')
ax8.text(0.76, 0.9,r'$M=8$',transform=ax8.transAxes,size=30, weight='bold')
ax9.text(0.76, 0.9,r'$M=9$',transform=ax9.transAxes,size=30, weight='bold')

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

ax6.set_xlabel(r"$t$")
ax6.set_ylabel(r"$\mathcal{E}$")
ax6.set_xticks([0.0,0.5,1.0,1.5])

ax7.set_xlabel(r"$t$")
ax7.set_ylabel(r"$\mathcal{E}$")
ax7.set_xticks([0.0,0.5,1.0,1.5])

ax8.set_xlabel(r"$t$")
ax8.set_ylabel(r"$\mathcal{E}$")
ax8.set_xticks([0.0,0.5,1.0,1.5])

ax9.set_xlabel(r"$t$")
ax9.set_ylabel(r"$\mathcal{E}$")
ax9.set_xticks([0.0,0.5,1.0,1.5])

# fig.suptitle(r"Turning off the magnetic field, $N=%s,h=%s,J=%s$, statevector simulation" %(n,-h,-J))

##### Close and save figure
ax5.legend(loc=(1.5,1.35))
# ax2.legend()
# ax3.legend()
# ax4.legend()
# plt.tight_layout()
plt.subplots_adjust(left=0.06,
                    bottom=0.09,
                    right=0.98,
                    top=0.98,
                    wspace=0.2,
                    hspace=0.3)
plt.savefig(dir_name+f"rxx_statevector_depthscale.eps", dpi=300)
plt.show()
#plt.close()





