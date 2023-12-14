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
passive_seeds = np.delete(passive_seeds,[52])
# print(passive_seeds)


m = 1           
depth_passs = [1,2]
dir_name_m = dir_name + f"m{m}/"
work_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
ergo_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name_m + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            ergotropy = quantities[3]
            ergo_list[j,i,k] = ergotropy
            total_work = quantities[1]
            work_list[j,i,k] = total_work
matlab_ergo_1 = []
matlab_time_1 = []
matlab_work_1 = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time_1.append(float(row[0]))
    matlab_ergo_1.append(float(row[2]))
    matlab_work_1.append(float(row[1]))
work_average_1 = np.average(work_list,axis=0)
ergo_average_1 = np.average(ergo_list,axis=0)
standard_deviation_ergo_1 = np.std(ergo_list,axis=0,ddof=1)

matlab_time_compare_1 = []
matlab_ergo_compare_1 = []
f=open(f"matlab_results_compare/matlab_data_N{n}_M{m}.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time_compare_1.append(float(row[0]))
    matlab_ergo_compare_1.append(float(row[2]))

abs_error_1 = np.zeros((reps+1,len(depth_passs)))
for k,depth_pass in enumerate(depth_passs):
    abs_error_1[:,k] = np.abs(matlab_ergo_compare_1-ergo_average_1[:,k])


m = 2           
depth_passs = [1,2]
dir_name_m = dir_name + f"m{m}/"
work_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
ergo_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name_m + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            ergotropy = quantities[3]
            ergo_list[j,i,k] = ergotropy
            total_work = quantities[1]
            work_list[j,i,k] = total_work
matlab_ergo_2 = []
matlab_time_2 = []
matlab_work_2 = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time_2.append(float(row[0]))
    matlab_ergo_2.append(float(row[2]))
    matlab_work_2.append(float(row[1]))
work_average_2 = np.average(work_list,axis=0)
ergo_average_2 = np.average(ergo_list,axis=0)
standard_deviation_ergo_2 = np.std(ergo_list,axis=0,ddof=1)
matlab_time_compare_2 = []
matlab_ergo_compare_2 = []
f=open(f"matlab_results_compare/matlab_data_N{n}_M{m}.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time_compare_2.append(float(row[0]))
    matlab_ergo_compare_2.append(float(row[2]))
abs_error_2 = np.zeros((reps+1,len(depth_passs)))
for k,depth_pass in enumerate(depth_passs):
    abs_error_2[:,k] = np.abs(matlab_ergo_compare_2-ergo_average_2[:,k])

m = 3           
depth_passs = [1,2,3]
dir_name_m = dir_name + f"m{m}/"
work_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
ergo_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name_m + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            ergotropy = quantities[3]
            ergo_list[j,i,k] = ergotropy
            total_work = quantities[1]
            work_list[j,i,k] = total_work
matlab_ergo_3 = []
matlab_time_3 = []
matlab_work_3 = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time_3.append(float(row[0]))
    matlab_ergo_3.append(float(row[2]))
    matlab_work_3.append(float(row[1]))
work_average_3 = np.average(work_list,axis=0)
ergo_average_3 = np.average(ergo_list,axis=0)
standard_deviation_ergo_3 = np.std(ergo_list,axis=0,ddof=1)
matlab_time_compare_3 = []
matlab_ergo_compare_3 = []
f=open(f"matlab_results_compare/matlab_data_N{n}_M{m}.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time_compare_3.append(float(row[0]))
    matlab_ergo_compare_3.append(float(row[2]))
abs_error_3 = np.zeros((reps+1,len(depth_passs)))
for k,depth_pass in enumerate(depth_passs):
    abs_error_3[:,k] = np.abs(matlab_ergo_compare_3-ergo_average_3[:,k])

m = 4           
depth_passs = [1,2,3]
dir_name_m = dir_name + f"m{m}/"
work_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
ergo_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name_m + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            ergotropy = quantities[3]
            ergo_list[j,i,k] = ergotropy
            total_work = quantities[1]
            work_list[j,i,k] = total_work
matlab_ergo_4 = []
matlab_time_4 = []
matlab_work_4 = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time_4.append(float(row[0]))
    matlab_ergo_4.append(float(row[2]))
    matlab_work_4.append(float(row[1]))
work_average_4 = np.average(work_list,axis=0)
ergo_average_4 = np.average(ergo_list,axis=0)
standard_deviation_ergo_4 = np.std(ergo_list,axis=0,ddof=1)
matlab_time_compare_4 = []
matlab_ergo_compare_4 = []
f=open(f"matlab_results_compare/matlab_data_N{n}_M{m}.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time_compare_4.append(float(row[0]))
    matlab_ergo_compare_4.append(float(row[2]))
abs_error_4 = np.zeros((reps+1,len(depth_passs)))
for k,depth_pass in enumerate(depth_passs):
    abs_error_4[:,k] = np.abs(matlab_ergo_compare_4-ergo_average_4[:,k])

m = 5           
depth_passs = [1,2,3]
dir_name_m = dir_name + f"m{m}/"
work_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
ergo_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name_m + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            ergotropy = quantities[3]
            ergo_list[j,i,k] = ergotropy
            total_work = quantities[1]
            work_list[j,i,k] = total_work
matlab_ergo_5 = []
matlab_time_5 = []
matlab_work_5 = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time_5.append(float(row[0]))
    matlab_ergo_5.append(float(row[2]))
    matlab_work_5.append(float(row[1]))
work_average_5 = np.average(work_list,axis=0)
ergo_average_5 = np.average(ergo_list,axis=0)
standard_deviation_ergo_5 = np.std(ergo_list,axis=0,ddof=1)
matlab_time_compare_5 = []
matlab_ergo_compare_5 = []
f=open(f"matlab_results_compare/matlab_data_N{n}_M{m}.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time_compare_5.append(float(row[0]))
    matlab_ergo_compare_5.append(float(row[2]))
abs_error_5 = np.zeros((reps+1,len(depth_passs)))
for k,depth_pass in enumerate(depth_passs):
    abs_error_5[:,k] = np.abs(matlab_ergo_compare_5-ergo_average_5[:,k])


m = 6           
depth_passs = [1,2,3,4]
dir_name_m = dir_name + f"m{m}/"
work_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
ergo_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name_m + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            ergotropy = quantities[3]
            ergo_list[j,i,k] = ergotropy
            total_work = quantities[1]
            work_list[j,i,k] = total_work
matlab_ergo_6 = []
matlab_time_6 = []
matlab_work_6 = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time_6.append(float(row[0]))
    matlab_ergo_6.append(float(row[2]))
    matlab_work_6.append(float(row[1]))
work_average_6 = np.average(work_list,axis=0)
ergo_average_6 = np.average(ergo_list,axis=0)
standard_deviation_ergo_6 = np.std(ergo_list,axis=0,ddof=1)
matlab_time_compare_6 = []
matlab_ergo_compare_6 = []
f=open(f"matlab_results_compare/matlab_data_N{n}_M{m}.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time_compare_6.append(float(row[0]))
    matlab_ergo_compare_6.append(float(row[2]))
abs_error_6 = np.zeros((reps+1,len(depth_passs)))
for k,depth_pass in enumerate(depth_passs):
    abs_error_6[:,k] = np.abs(matlab_ergo_compare_6-ergo_average_6[:,k])

m = 7           
depth_passs = [1,2,3,4]
dir_name_m = dir_name + f"m{m}/"
work_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
ergo_list = np.zeros((len(passive_seeds),reps+1,len(depth_passs)))
for k,depth_pass in enumerate(depth_passs):
    for i,index_rep in enumerate(index_reps):
        for j,passive_seed in enumerate(passive_seeds):
            dir_name_depth = dir_name_m + f"depth_pass{depth_pass}/"
            dir_name_time = dir_name_depth + f"time_index{index_rep}/"
            save_name_quantities = dir_name_time + f"quantities_seed{passive_seed}"
            quantities = np.load(save_name_quantities+'.npy')
            ergotropy = quantities[3]
            ergo_list[j,i,k] = ergotropy
            total_work = quantities[1]
            work_list[j,i,k] = total_work
matlab_ergo_7 = []
matlab_time_7 = []
matlab_work_7 = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time_7.append(float(row[0]))
    matlab_ergo_7.append(float(row[2]))
    matlab_work_7.append(float(row[1]))
work_average_7 = np.average(work_list,axis=0)
ergo_average_7 = np.average(ergo_list,axis=0)
standard_deviation_ergo_7 = np.std(ergo_list,axis=0,ddof=1)
matlab_time_compare_7 = []
matlab_ergo_compare_7 = []
f=open(f"matlab_results_compare/matlab_data_N{n}_M{m}.txt",'r')
for row in f:
    row=row.split(' ')
    matlab_time_compare_7.append(float(row[0]))
    matlab_ergo_compare_7.append(float(row[2]))
abs_error_7 = np.zeros((reps+1,len(depth_passs)))
for k,depth_pass in enumerate(depth_passs):
    abs_error_7[:,k] = np.abs(matlab_ergo_compare_7-ergo_average_7[:,k])


###################################################### Begin plotting Fig.4 #######################################################
plt.rc('font', family='serif')#, serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=35)
plt.rc('ytick', labelsize=35)
plt.rc('axes', labelsize=35)
plt.rc('legend', fontsize=35)
plt.rc('legend', handlelength=2)
plt.rc('font', size=35)
linestyles = ["-", "--", "dotted"]
colors = ['tab:orange','tab:blue', 'tab:red', 'tab:brown','tab:purple', 'tab:green', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
markers = ["o", "v", "s", "d", "*"]
##### Specify columns and rows
fig = plt.figure(figsize=(28,12))
gs = fig.add_gridspec(nrows=2,ncols=8) #height_ratios=[1,1], width_ratios=[1,1])
ax1 = fig.add_subplot(gs[0,0:2])
ax2 = fig.add_subplot(gs[0,2:4])
ax3 = fig.add_subplot(gs[0,4:6])
ax4 = fig.add_subplot(gs[0,6:8])
ax5 = fig.add_subplot(gs[1,0:2])
ax6 = fig.add_subplot(gs[1,2:4])
ax7 = fig.add_subplot(gs[1,4:6])


ax1.plot(matlab_time_1, matlab_work_1, ls=linestyles[1],lw=3,label="Work",color = 'dimgrey')
ax2.plot(matlab_time_2, matlab_work_2, ls=linestyles[1],lw=3,label="Work",color = 'dimgrey')
ax3.plot(matlab_time_3, matlab_work_3, ls=linestyles[1],lw=3,label="Work",color = 'dimgrey')
ax4.plot(matlab_time_4, matlab_work_4, ls=linestyles[1],lw=3,label="Work",color = 'dimgrey')
ax5.plot(matlab_time_5, matlab_work_5, ls=linestyles[1],lw=3,label="Work",color = 'dimgrey')
ax6.plot(matlab_time_6, matlab_work_6, ls=linestyles[1],lw=3,label="Work",color = 'dimgrey')
ax7.plot(matlab_time_7, matlab_work_7, ls=linestyles[1],lw=3,label="Work",color = 'dimgrey')

# ax1.plot(times, work_average_1[:,1], ls='',lw=3,marker="*",ms=6,label="SV-Work",color = 'cyan')
# ax2.plot(times, work_average_2[:,1], ls='',lw=3,marker="*",ms=6,label="SV-Work",color = 'cyan')
# ax3.plot(times, work_average_3[:,1], ls='',lw=3,marker="*",ms=6,label="SV-Work",color = 'cyan')
# ax4.plot(times, work_average_4[:,1], ls='',lw=3,marker="*",ms=6,label="SV-Work",color = 'cyan')
# ax5.plot(times, work_average_5[:,1], ls='',lw=3,marker="*",ms=6,label="SV-Work",color = 'cyan')
# ax6.plot(times, work_average_6[:,1], ls='',lw=3,marker="*",ms=6,label="SV-Work",color = 'cyan')
# ax7.plot(times, work_average_7[:,1], ls='',lw=3,marker="*",ms=6,color = 'cyan')

ax1.plot(matlab_time_1, matlab_ergo_1, ls=linestyles[0],lw=3,label="ED",color = colors[0])
ax2.plot(matlab_time_2, matlab_ergo_2, ls=linestyles[0],lw=3,label="ED",color = colors[0])
ax3.plot(matlab_time_3, matlab_ergo_3, ls=linestyles[0],lw=3,label="ED",color = colors[0])
ax4.plot(matlab_time_4, matlab_ergo_4, ls=linestyles[0],lw=3,label="ED",color = colors[0])
ax5.plot(matlab_time_5, matlab_ergo_5, ls=linestyles[0],lw=3,label="ED",color = colors[0])
ax6.plot(matlab_time_6, matlab_ergo_6, ls=linestyles[0],lw=3,label="ED",color = colors[0])
ax7.plot(matlab_time_7, matlab_ergo_7, ls=linestyles[0],lw=3,label="ED",color = colors[0])

depth_pass_1 =[1,2]
depth_pass_2 =[1,2,3]
depth_pass_3 =[1,2,3,4]
for k,depth_pass in enumerate(depth_pass_1):
    ax1.errorbar(times, ergo_average_1[:,k], yerr=standard_deviation_ergo_1[:,k],capsize =6 ,ms=8, marker = markers[k],ls='',label=f"depth = {depth_pass}",color = colors[k+1])
    ax2.errorbar(times, ergo_average_2[:,k], yerr=standard_deviation_ergo_2[:,k],capsize =6 ,ms=8, marker = markers[k],ls='',label=f"depth = {depth_pass}",color = colors[k+1])

for k,depth_pass in enumerate(depth_pass_2):    
    ax3.errorbar(times, ergo_average_3[:,k], yerr=standard_deviation_ergo_3[:,k],capsize =6 ,ms=8, marker = markers[k],ls='',color = colors[k+1])
    ax4.errorbar(times, ergo_average_4[:,k], yerr=standard_deviation_ergo_4[:,k],capsize =6 ,ms=8, marker = markers[k],ls='',label=f"depth = {depth_pass}",color = colors[k+1])
    ax5.errorbar(times, ergo_average_5[:,k], yerr=standard_deviation_ergo_5[:,k],capsize =6 ,ms=8, marker = markers[k],ls='',label=f"depth = {depth_pass}",color = colors[k+1])

for k,depth_pass in enumerate(depth_pass_3):
    ax6.errorbar(times, ergo_average_6[:,k], yerr=standard_deviation_ergo_6[:,k],capsize =6 ,ms=8, marker = markers[k],ls='',color = colors[k+1])
    ax7.errorbar(times, ergo_average_7[:,k], yerr=standard_deviation_ergo_7[:,k],capsize =6 ,ms=8, marker = markers[k],ls='',color = colors[k+1],label=f"depth = {depth_pass}")


ax1.text(0.05, 0.9,'a)',transform=ax1.transAxes,size=35, weight='bold')
ax2.text(0.05, 0.9,'b)',transform=ax2.transAxes,size=35, weight='bold')
ax3.text(0.05, 0.9,'c)',transform=ax3.transAxes,size=35, weight='bold')
ax4.text(0.05, 0.9,'d)',transform=ax4.transAxes,size=35, weight='bold')
ax5.text(0.05, 0.9,'e)',transform=ax5.transAxes,size=35, weight='bold')
ax6.text(0.05, 0.9,'f)',transform=ax6.transAxes,size=35, weight='bold')
ax7.text(0.05, 0.9,'g)',transform=ax7.transAxes,size=35, weight='bold')


ax1.text(0.72, 0.91,r'$M=1$',transform=ax1.transAxes,size=35, weight='bold')
ax2.text(0.72, 0.91,r'$M=2$',transform=ax2.transAxes,size=35, weight='bold')
ax3.text(0.72, 0.91,r'$M=3$',transform=ax3.transAxes,size=35, weight='bold')
ax4.text(0.72, 0.91,r'$M=4$',transform=ax4.transAxes,size=35, weight='bold')
ax5.text(0.72, 0.91,r'$M=5$',transform=ax5.transAxes,size=35, weight='bold')
ax6.text(0.72, 0.91,r'$M=6$',transform=ax6.transAxes,size=35, weight='bold')
ax7.text(0.72, 0.91,r'$M=7$',transform=ax7.transAxes,size=35, weight='bold')

ax1.set_xlabel(r"$t$")
ax1.set_ylabel(r"$\mathcal{E}$")
ax1.set_xticks([0.0, 0.4, 0.8, 1.2])

ax2.set_xlabel(r"$t$")
ax2.set_ylabel(r"$\mathcal{E}$")
ax2.set_xticks([0.0, 0.4, 0.8, 1.2])
ax2.set_ylim([0,1.52])

ax3.set_xlabel(r"$t$")
ax3.set_ylabel(r"$\mathcal{E}$")
ax3.set_xticks([0.0, 0.4, 0.8, 1.2])
ax3.set_ylim([0,1.9])

ax4.set_xlabel(r"$t$")
ax4.set_ylabel(r"$\mathcal{E}$")
ax4.set_xticks([0.0, 0.4, 0.8, 1.2])

ax5.set_xlabel(r"$t$")
ax5.set_ylabel(r"$\mathcal{E}$")
ax5.set_xticks([0.0, 0.4, 0.8, 1.2])

ax6.set_xlabel(r"$t$")
ax6.set_ylabel(r"$\mathcal{E}$")
ax6.set_xticks([0.0, 0.4, 0.8, 1.2])

ax7.set_xlabel(r"$t$")
ax7.set_ylabel(r"$\mathcal{E}$")
ax7.set_xticks([0.0, 0.4, 0.8, 1.2])
# fig.suptitle(r"$N=%s,h=%s,J=%s$, statevector simulation" %(n,-h,-J))

##### Close and save figure
# ax2.legend(loc=0)
ax7.legend(bbox_to_anchor = (1.3, 1.0))
# plt.tight_layout()
plt.subplots_adjust(left=0.05,
                    bottom=0.09,
                    right=0.98,
                    top=0.98,
                    wspace=0.7,
                    hspace=0.3)
# plt.subplots_adjust(right=1.2)
plt.savefig(dir_name+f"pvqd_statevector_depthscale_ver2.eps",bbox_inches='tight', dpi=300)
plt.show()

#plt.close()



# ####################################################### Begin plotting Fig.5 #######################################################
# plt.rc('font', family='serif')#, serif='Times')
# plt.rc('text', usetex=True)
# plt.rc('xtick', labelsize=35)
# plt.rc('ytick', labelsize=35)
# plt.rc('axes', labelsize=35)
# plt.rc('legend', fontsize=30)
# plt.rc('legend', handlelength=2)
# plt.rc('font', size=35)
# linestyles = ["-", "--", "dotted"]
# colors = ['tab:orange','tab:blue', 'tab:red', 'tab:brown','tab:purple', 'tab:green', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
# markers = ["o", "v", "s", "d", "*"]
# ##### Specify columns and rows
# fig = plt.figure(figsize=(13,14))
# gs = fig.add_gridspec(nrows=2,ncols=1) #height_ratios=[1,1], width_ratios=[1,1])
# ax1 = fig.add_subplot(gs[0,0])
# ax2 = fig.add_subplot(gs[1,0])

# M_list = [1,2,3,4,5,6,7]
# depth_pass_1 =[1,2]
# depth_pass_2 =[1,2,3]
# depth_pass_3 =[1,2,3,4]
# for k,depth_pass in enumerate(depth_pass_1):
#     ax1.errorbar(M_list[0],abs_error_1[2,k],yerr=standard_deviation_ergo_1[2,k],capsize =6, ms=8, marker = markers[k],ls='',color = colors[k+1])
#     ax1.errorbar(M_list[1],abs_error_2[2,k],yerr=standard_deviation_ergo_2[2,k],capsize =6, ms=8, marker = markers[k],ls='',color = colors[k+1])
#     ax2.errorbar(M_list[0],abs_error_1[4,k],yerr=standard_deviation_ergo_1[4,k],capsize =6, ms=8, marker = markers[k],ls='',color = colors[k+1])
#     ax2.errorbar(M_list[1],abs_error_2[4,k],yerr=standard_deviation_ergo_2[4,k],capsize =6, ms=8, marker = markers[k],ls='',color = colors[k+1])
# for k,depth_pass in enumerate(depth_pass_2):
#     ax1.errorbar(M_list[2],abs_error_3[2,k],yerr=standard_deviation_ergo_3[2,k],capsize =6, ms=8, marker = markers[k],ls='',color = colors[k+1])
#     ax1.errorbar(M_list[3],abs_error_4[2,k],yerr=standard_deviation_ergo_4[2,k],capsize =6, ms=8, marker = markers[k],ls='',color = colors[k+1])
#     ax1.errorbar(M_list[4],abs_error_5[2,k],yerr=standard_deviation_ergo_5[2,k],capsize =6, ms=8, marker = markers[k],ls='',color = colors[k+1])
#     ax2.errorbar(M_list[2],abs_error_3[4,k],yerr=standard_deviation_ergo_3[4,k],capsize =6, ms=8, marker = markers[k],ls='',color = colors[k+1])
#     ax2.errorbar(M_list[3],abs_error_4[4,k],yerr=standard_deviation_ergo_4[4,k],capsize =6, ms=8, marker = markers[k],ls='',color = colors[k+1])
#     ax2.errorbar(M_list[4],abs_error_5[4,k],yerr=standard_deviation_ergo_5[4,k],capsize =6, ms=8, marker = markers[k],ls='',color = colors[k+1])
# for k,depth_pass in enumerate(depth_pass_3):
#     ax1.errorbar(M_list[5],abs_error_6[2,k],yerr=standard_deviation_ergo_6[2,k],capsize =6, ms=8, marker = markers[k],ls='',label=f"depth = {depth_pass}",color = colors[k+1])
#     ax1.errorbar(M_list[6],abs_error_7[2,k],yerr=standard_deviation_ergo_7[2,k],capsize =6, ms=8, marker = markers[k],ls='',color = colors[k+1])
#     ax2.errorbar(M_list[5],abs_error_6[4,k],yerr=standard_deviation_ergo_6[4,k],capsize =6, ms=8, marker = markers[k],ls='',color = colors[k+1])
#     ax2.errorbar(M_list[6],abs_error_7[4,k],yerr=standard_deviation_ergo_7[4,k],capsize =6, ms=8, marker = markers[k],ls='',color = colors[k+1])

# ax1.text(0.05, 0.9,'a) t = 0.4',transform=ax1.transAxes,size=35, weight='bold')
# ax2.text(0.05, 0.9,'b) t = 0.8',transform=ax2.transAxes,size=35, weight='bold')

# # ax1.set_xlabel(r"$M$")
# ax1.set_ylabel(r"$|\Delta \mathcal{E}|$")

# ax2.set_xlabel(r"$M$")
# ax2.set_ylabel(r"$|\Delta \mathcal{E}|$")
# ax1.tick_params('both', length=10, width=2, which='major')
# ax2.tick_params('both', length=10, width=2, which='major')
# ax1.legend(loc=4)
# plt.tight_layout()
# plt.savefig(dir_name+f"pvqd_statevector_abs_error.eps", dpi=300)
# plt.show()