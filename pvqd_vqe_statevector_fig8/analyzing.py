import numpy as np 
import pickle
import matplotlib.pyplot as plt
import itertools
####################################################### Parameters of the systems #######################################################
n = 4           # number of qubits (total cells)
m = 2           # number of cells we want to extract the energy (M <= N)
h  = -0.6       # a transverse magnetic field
J  = -2.0       # J is nearest-neighbor interaction strength

####################################################### Optimum parameters for time evolution using PVQD #######################################################
depth = 2       # depth of the ansatz describing the evolved-state
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

# error_work = np.zeros((len(passive_seeds),reps+1))
# error_ergo = np.zeros((len(passive_seeds),reps+1))
work_list = np.zeros((len(passive_seeds),reps+1))
ergo_list = np.zeros((len(passive_seeds),reps+1))

####################################################### Load data of matlab #######################################################
##### N2_M1_J4
# total_work_matlab = [0, 0.0473064427761583, 0.181094736152413, 0.378369338172935, 0.605222675363978, 0.822663179019858, 0.993317158972830, 1.08785259627785, 1.09002073284106, 0.999448909186451, 0.831704617194718,0.615619758424676,0.388335014698849,0.188916104910658,0.0516391663146790,9.93694702213777e-05,0.0431553785460441,0.173406723454909,0.368465793210318,0.594805820191349,0.813523463489926]
# ergotropy_matlab = [0,0,0,0,0.0104453507279569,0.445326358039716,0.786634317945660,0.975705192555693,0.980041465682111,0.798897818372902,0.463409234389435,0.0312395168493514,0,0,0,0,0,0,0,0,0.427046926979853]

##### N2_M1_J2
# total_work_matlab = [0,0.0471358888288627,0.178471431511003,0.365942495202304,0.569489757586686,0.745618690869516,0.856693589739056,0.878979670344248,0.807714782897564,0.658127000680472,0.462180644289395,0.261746059434094,0.0996526485829374,0.0105369687743537,0.0134414987278692]
# ergotropy_matlab = [0,0,0,0,0,0.291237381739033,0.513387179478111,0.557959340688496,0.415429565795129,0.116254001360943,0,0,0,0,0]

##### N4_M1
# total_work_matlab = [0,0.0471389024931980,0.178653511558502,0.367826181711813,0.578734015320143,0.775220991991495,0.927942827191740,1.01786133683271,1.03668494759454,0.985937730111310,0.876017262674090,0.725330221964796,0.558548414420335,0.403077679167848,0.283922664119026]
# ergotropy_matlab = [0,0,0,0,0,0.350441983982989,0.655885654383480,0.835722673665425,0.873369895189079,0.771875460222619,0.552034525348179,0.250660443929592,0,0,0]

##### N4_M2
total_work_matlab = [0,0.137692802859178,0.481642020280141,0.868092948670199,1.13951814150673,1.23411507493769,1.20564922064944,1.16593797096761,1.19581788123833,1.28874849529406,1.36318930549216,1.32850425836185,1.15289792997237,0.885628473450649,0.623213843819488]
ergotropy_matlab = [0,0.0905509633358443,0.302824311369582,0.498773733400337,0.554624753583859,0.823958963661795,0.958867488622713,1.01757239788005,1.07137375393794,1.11980350692653,1.09832469495944,0.935044103367898,0.611243602813555,0.289770264732029,0.121038914867986]

##### N4_M3
# total_work_matlab = [0,0.228246703225160,0.784630529001781,1.36835971562858,1.70030226769332,1.69300915788388,1.48335561410713,1.31401460510251,1.35495081488211,1.59155926047682,1.85036134831022,1.93167829475891,1.74724744552440,1.36817926773345,0.962505023519950]
# ergotropy_matlab = [0,0.181107800731962,0.605977017443279,1.00053353391677,1.12156825237318,1.26823014987537,1.21129844129888,1.13187594193522,1.19163576247665,1.37749699058813,1.52637861098431,1.45700851672371,1.18869903110407,0.965101588565603,0.678582359400924]

##### N6_M2
# total_work_matlab = [0,0.481642036375405,1.13952953937969,1.20598513415406,1.19773090388958,1.36400628456420,1.13881320319176,0.601011683213783]
# ergotropy_matlab = [0,0.302824313053119,0.554629060998684,0.959221941958188,1.07189936451931,1.09272409942696,0.596870383282357,0.118767812632324]

##### N6_M3
# total_work_matlab = [0,0.784794218689171,1.70601796501561,1.49936080326893,1.31507266008584,1.66660529947968,1.53859027435256,0.956721455493455]
# ergotropy_matlab = [0,0.605976498925970,1.12116045313206,1.25356300858832,1.19091096025544,1.38795811306439,0.976163808042590,0.525637168049340]

##### N6_M4
# total_work_matlab = [0,1.08794640100294,2.27250639065154,1.79273647238379,1.43241441628211,1.96920431439516,1.93836734551335,1.31243122777313]
# ergotropy_matlab = [0,0.909128695646854,1.68765578909556,1.54691560555288,1.30986763630910,1.70139154357558,1.40103909047499,0.852237260802175]

##### N8_M2
# total_work_matlab = [0,0.481642036375405,1.13952953938022,1.20598513470047,1.19773096081156,1.36400780289713,1.13882804604003,0.601066512938263]
# ergotropy_matlab = [0,0.302824313053120,0.554629060999015,0.959221942318547,1.07189937404310,1.09272419237168,0.596876794067286,0.118887808763979]

##### N8_M4
# total_work_matlab = [0,1.08794641709817,2.27251778709182,1.79307170658725,1.43428501161356,1.96921516357674,1.91763969376501,1.26242249226549]
# ergotropy_matlab = [0,0.909128697334387,1.68766027139151,1.54727360349151,1.31008600559522,1.69003318524258,1.35287134420414,0.839809928625580]

##### N8_M6
# total_work_matlab = [-4.44089209850063e-16,1.69425079782094,3.40550603480342,2.38015827847403,1.67083906241556,2.57442252425636,2.69645134148999,1.92377847159271]
# ergotropy_matlab = [0,1.51543309246486,2.82065543324725,2.13433741148368,1.54829223765880,2.30660825923717,2.15911110350353,1.46363560081243]

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
matlab_ergo = []
matlab_time = []
matlab_work = []
f=open(f"matlab_results/matlab_data_N{n}_M{m}.txt",'r')
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
# variance_ergo = np.var(ergo_list,axis=0)
# variance_work = np.var(work_list,axis=0)
standard_deviation_ergo = np.std(ergo_list,axis=0,ddof=1)
standard_deviation_work = np.std(work_list,axis=0,ddof=1)
standard_error_ergo     = standard_deviation_ergo/np.sqrt(len(passive_seeds))
standard_error_work     = standard_deviation_work/np.sqrt(len(passive_seeds))
error_work = abs(average_work - total_work_matlab)
error_ergo = abs(average_ergo - ergotropy_matlab)

ax1.plot(matlab_time, matlab_work, ls=linestyles[0],lw=3,color = colors[1],label="ED")
ax1.errorbar(times, average_work, yerr=standard_error_work,capsize =6 ,ms=6, marker = "v",ls='',label="VQE - depth = 2",color = colors[3])
ax2.plot(times,standard_deviation_work, ms=6, marker="x",color = colors[3],ls='')
ax3.plot(times, error_work, ms=6, marker="x",color = colors[3],ls='')

ax4.plot(matlab_time, matlab_ergo, ls=linestyles[0],lw=3,color = colors[1],label="ED")
ax4.errorbar(times, average_ergo, yerr=standard_error_ergo,capsize =6 ,ms=6, marker = "v",ls='',label="VQE - depth = 2",color = colors[3])
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
ax1.set_xticks([0.0, 0.4, 0.8, 1.2])

ax2.set_xlabel(r"$t$")
ax2.set_ylabel(r"$\sigma(W)$")
ax2.set_yscale('log')
ax2.set_xticks([0.0, 0.4, 0.8, 1.2])

ax3.set_xlabel(r"$t$")
ax3.set_ylabel(r"$|\Delta W|$")
#ax3.set_yscale('log')
ax3.set_xticks([0.0, 0.4, 0.8, 1.2])

ax4.set_xlabel(r"$t$")
ax4.set_ylabel(r"$\mathcal{E}$")
ax4.set_xticks([0.0, 0.4, 0.8, 1.2])

ax5.set_xlabel(r"$t$")
ax5.set_ylabel(r"$\sigma (\mathcal{E})$")
ax5.set_yscale('log')
ax5.set_xticks([0.0, 0.4, 0.8, 1.2])

ax6.set_xlabel(r"$t$")
ax6.set_ylabel(r"$|\Delta \mathcal{E}|$")
#ax6.set_yscale('log')
ax6.set_xticks([0.0, 0.4, 0.8, 1.2])

fig.suptitle(r"Keeping the magnetic field, $N=%s,M=%s,h=%s,J=%s$, without noise" %(n,m,-h,-J))

##### Close and save figure
ax1.legend(loc=8)
# ax2.legend()
# ax3.legend()
# ax4.legend()
plt.tight_layout()
plt.savefig(dir_name+f"pvqd_statevector_N{n}_M{m}.eps", dpi=300)
plt.show()
#plt.close()




