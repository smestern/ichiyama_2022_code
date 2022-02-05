
import os
from brian2 import *
from utils import *
from loadNWB import *
import pandas as pd
import numpy as np

import torch
from pickle import dump, load
from sbi import utils as utils
from sbi.inference.base import infer
from sbi.utils.get_nn_models import posterior_nn
from sbi.inference import SNPE, prepare_for_sbi
print("== Loading Data ==")
file_dir = os.path.dirname(os.path.realpath(__file__))
default_dtype = torch.float32

#%% Load and/or compute fixed params

#file_path = file_dir +'//..//data//HYP_CELL_NWB//Naive//SM-A1-C11(111)-P08-16d05019.nwb'
file_path = file_dir +'//..//data//HYP_CELL_NWB//Naive//SM-A1-C11(111)-P08-16624010.nwb'
realX, realY, realC, _ = loadNWB(file_path)
n=9; realX, realY, realC = realX[:n,:], realY[:n,:], realC[:n,:] #Take only the first 9 sweeps
spike_times = detect_spike_times(realX, realY, realC, speak=-10) # spkIdxs = spikeIndices(V)


real_fi, real_isi = compute_FI_curve(spike_times, 2) #Compute the real FI curve
#real_isi *= 10 #
real_fi = np.hstack((real_fi, real_isi))
real_rmp = compute_rmp(realY, realC)
real_min = []
real_subt = []
for x in np.arange(realX.shape[0]):
    temp = compute_steady_hyp(realY[x, :].reshape(1,-1), realC[x, :].reshape(1,-1))
    temp_min = compute_min_stim(realY[x, :], realX[x,:], strt=0.62, end=1.2)
    real_subt.append(temp)
    real_min.append(temp_min)
real_subt = np.nanmean(real_subt)

Cell_params = {'Cm': 11.70570803*pF, 
'EL': -67.44385*mV,
'taum': 16.3721611190163*ms}
dt = (realX[0,1] - realX[0,0]) * 1000

#%% crop the data to the first 2seconds
realX = realX[:,:20000]
realC = realC[:,:20000]
realY = realY[:,:20000]

TotTime=2000
start_scope()

#integration step
defaultclock.dt = dt*ms

#simulation duration
    
duration = TotTime*ms

#number of neuron
print("== Setting up model ==")

#%%
def adex_model(cm_i, taum_i, El_i, a_i, b_i, tau_a_i, Dt_i, Vt_i, VR_i, refrac, realC_i=None, record_v=False) -> [StateMonitor, SpikeMonitor]:
    '''
    Simple adex Model function that takes param inputs and outputs the voltage and spike times
    ---
    Takes:
    For below the inputs are in array shape (num_units,), where num_units is the number of realizations of the simulation
    cm_i (numpy array): Cell Capacitance (cm) in picofarad 
    taum_i (numpy array): Cell time Constant in ms
    El_i (numpy array): Leak potential in mV
    a_i (numpy array): Max subthreshold conductance in nano-siemens
    b_i (numpy array): Spike-triggered adaptation Conductance in nano-siemens
    tau_a_i (numpy array): Time constant for adaptation in ms
    Dt_i (numpy array): DeltaT, exponential for AdEx in mV
    Vt_i (numpy array): Voltage threshold for registering a spike in mV
    VR_i (numpy array): Reset potential for Vm post-spike in mV

    realC_i (numpy array): 1D numpy array representing the input current in picoamp
    record_v (bool): Whether to record the voltage for all cells (true), or not (false)

    Returns:
    voltage (brian2 voltage monitor) with shape (num_units, time_steps)
    spike times (brian2 spike monitor)

    '''
    start_scope()
    eqs='''
    dv/dt = ( gL*(EL-v) + gL*DeltaT*exp((v - VT)/DeltaT) + I - w ) * (1./Cm) : volt  (unless refractory)
        dw/dt = ( a*(v - EL) - w ) / tauw : amp
        tauw : second
        a : siemens
        b : amp
        Cm : farad
        taum : second
        gL : siemens
        EL : volt
        VT : volt
        VR : volt
        Vcut : volt
        DeltaT : volt
        refrac : second
    I = in_current(t) : amp
    '''
    in_current = TimedArray(values = realC_i * pamp, dt=dt * ms)
    G1 = NeuronGroup(N, eqs, threshold='v > Vcut', reset='v = VR; w += b', method='Euler', refractory='refrac')
    #init:
    G1.v = El_i *mV
    G1.a = a_i * nS
    G1.b = b_i * pA
    G1.tauw = tau_a_i * ms
    G1.DeltaT = (Dt_i) *mV
    G1.VT = Vt_i * mV
    G1.VR = VR_i * mV
    #parameters
    G1.Cm = cm_i * pF
    G1.gL = ((cm_i *pF)/ (taum_i * ms))
    G1.EL = El_i *mV
    G1.Vcut = (Vt_i + 5 * Dt_i)*mV
    G1.refrac = refrac*ms
    # record variables
    if record_v == True:
        #Only record voltage if explicity asked to save memory
        Mon_v = StateMonitor( G1, "v", record=True, dt=dt * ms)
    else:
        Mon_v = None
    Mon_spike = SpikeMonitor( G1 )
    run(duration)
    return Mon_v, Mon_spike

def generate_fi_curve(param_set, save=True) -> torch.tensor:
    ''' Function to handle interfacing between SNPE and brian2, running each sweep and computing the FI curve.
    Takes:
    param_set (torch.tensor) of shape (num_units, params): A tensor containg the randomly sampled params, each row represents a single cell param set.

    Returns:
    vars (torch.tensor) of shape (num_units, measured vars): A tensor containing the FI curve, ISI-mode-Curve (most common ISI per Sweep), and subthreshold params
    '''
    param_list = param_set.numpy() #Input is pytorch tensor converting it to a numpy array for brian2 purposes
    if param_list.shape[0] > 12:
        cm_i, taum_i, El_i, a_i, Dga_i, tau_a_i, Dt_i, Vt_i, VR_i, refrac = np.hsplit(np.vstack(param_list), 10) #Splits the variables out of the array
        ## Then flatten the variable arrays
        cm_i, taum_i, El_i, a_i, Dga_i, tau_a_i, Dt_i, Vt_i, VR_i, refrac = np.ravel(cm_i), np.ravel(taum_i), np.ravel(El_i), np.ravel(a_i), np.ravel(Dga_i), np.ravel(tau_a_i), np.ravel(Dt_i), np.ravel(Vt_i), np.ravel(VR_i), np.ravel(refrac)
    else:
        cm_i, taum_i, El_i, a_i, Dga_i, tau_a_i, Dt_i, Vt_i, VR_i, refrac = param_list
    spikes_full = [[] for x in np.arange(N)]
    isi_full = [[] for x in np.arange(N)]
    ##Generate an empty list of length N_UNITS which allows us to dump the subthreshold params into
    subthres_features = [[] for x in np.arange(N)]
    stim_min = [[] for x in np.arange(N)]
    for i, sweep in enumerate(realC): #For each sweep
        print(f"Simulating sweep {i}")
        voltage, spikes = adex_model(cm_i, taum_i, El_i, a_i, Dga_i, tau_a_i, Dt_i, Vt_i, VR_i, refrac, realC_i=sweep, record_v=True) #Run the adex model with the passed in params
        temp_spike_array = spikes.spike_trains() # Grab the spikes oriented by neuron in the network
        print("Simulation Finished")
        for p in np.arange(N): #For each neuron
                pspikes = temp_spike_array[p] #Grab that neurons spike times
                if len(pspikes) > 0: #If there are any spikes
                    spikes_full[p].append(len(pspikes)) #Count the number of spikes and add it to the full array
                    spike_s = pspikes/ms #get the spike time in ms
                    if len(spike_s) > 1: #If there is more than one spike
                        isi_full[p].append(np.nanmean(np.diff(spike_s))) #compute the mode ISI
                    else:
                        isi_full[p].append(0) #otherwise the mode ISI is set to zero
                else:
                    spikes_full[p].append(0) #If no spikes then send both to zero
                    isi_full[p].append(0)

                ##Compute Subthresfeatures
                temp_rmp = compute_rmp(voltage[p].v/mV.reshape(1,-1), sweep.reshape(1,-1)) #compute the beginning Resting membrane
                temp_deflection = compute_steady_hyp(voltage[p].v/mV.reshape(1,-1), sweep.reshape(1,-1)) #compute the end
                subthres_features[p].append(np.hstack((temp_rmp, temp_deflection)))

                #compute Sweepwisemin
                temp_min = compute_min_stim(voltage[p].v/mV, voltage[0].t/second, strt=0.62, end=1.2)
                stim_min[p].append(temp_min)
        
    neuron_avgs = np.vstack([np.nanmean(np.vstack(x), axis=0) for x in subthres_features]) #For each neuron, compute the mean accross sweeps
    #of the SubT features
    spikes_return = np.array(spikes_full) #Stack all the arrays together
    isi_return = np.array(isi_full) #Stack all the arrays together
    min_return = np.array(stim_min)
    return_full = np.hstack((spikes_return / 2, isi_return, neuron_avgs, min_return))
    #Load and save array
    if save:
        if os.path.exists("theta_ds.npy"):
            out_return_full = np.load("theta_ds.npy")
            out_params = np.load("params_ds.npy")
            out_return_full = np.vstack((out_return_full, return_full))
            out_params = np.vstack((out_params, np.vstack(param_list)))
        else:
            out_return_full = return_full
            out_params = np.vstack(param_list)
        np.save("params_ds.npy", out_params)
        np.save("theta_ds.npy", out_return_full)
    return torch.tensor(return_full[:, :22], dtype=default_dtype)


#%% Grid Params ##
##Global vars ###
N = 50000
batches = 1
#Ranges in lower -> upper
_cm_range = [2, 25] #in pF
_taum_range = [2, 30] #in ms
_EL_range = [-90, -50] #inmV
_VT_range = [-60., -30.] #in mV
_tauw_range = [0.01, 900.] #in ms
_a_range= [0.000001, 2] #in ns,
_b_range = [0.00001, 50] #in nS
_DeltaT_range =[0.1,15] #in mV
_VR_range = [-80., -30.] #in mV
_refrac = [0, 2] #in mS
vars = [_cm_range,
        _taum_range,
        _EL_range,
        _a_range, 
        _b_range,
        _tauw_range, 
        _DeltaT_range ,
        _VT_range,
        _VR_range,
        _refrac
        ]
def vars_to_param_space():
    vars_stack = np.vstack(vars) #Stack the grid ranges on top of each other
    lower_bound = vars_stack[:,0] #First column is lower bound
    upper_bound = vars_stack[:,1] #Second column is upper bound
    prior = utils.BoxUniform(low=torch.tensor(lower_bound, dtype=default_dtype), high=torch.tensor(upper_bound, dtype=default_dtype)) #The SNPE-SBI wants the data in torch tensors which
    #are similar to numpy arrays
    return prior

prior = vars_to_param_space() #Converts our ranges into the format used by SBI
simulator = generate_fi_curve

##Now intialize the Neural Network. We tell it to run with a batch size same as the number of neurons we simulate in parallel
## from https://elifesciences.org/articles/56261j
#on

density = posterior_nn(model='maf', hidden_features=100, num_transforms=10) #this is the same as the default method however a bit buffered to account for size
inference = SNPE(simulator, prior, device="cpu", num_workers=1, simulation_batch_size=N, logging_level='INFO')
#%% Run the inference
load_infer = False
load_data = False
print("== Fitting Model ==")
#Now run the inference for N of neuron simulations (1 batch). This runs the simulator function we provid with randomly selected params
#and then computes the prob dist
if load_infer == True:
    with open("adexpost.pkl", "rb") as f:
        posterior = load(f)
    theta = np.load("theta_ds.npy")
    _params = np.load("params_ds.npy")
elif load_data == True:
    theta = np.load("theta_ds.npy") 
    _params = np.load("params_ds.npy")
    inference.provide_presimulated(torch.tensor(_params, dtype=torch.float32), torch.tensor(theta, dtype=torch.float32))
    posterior = inference(num_simulations=0)
else:
    posterior = inference(num_simulations=N * batches)
    with open("adexpost.pkl", "wb") as f:
        dump(posterior, f)
    theta = np.load("theta_ds.npy")
    _params = np.load("params_ds.npy")
    
#%% Draw from the distro      
print("== Fitting Spikes ==")
##Now provide the real FI observation
x_o = torch.tensor(np.hstack((real_fi, real_rmp, real_subt, real_min))[:22], dtype=default_dtype)

#Find min
def abs_min(x1, x2):
    y = np.sum(np.square(x1 - x2))
    return y
x_test = x_o.numpy()
min_idx = np.nan_to_num(np.apply_along_axis(abs_min, 1, theta[:,:22], x_test), nan=900000000)
print(np.amin(min_idx))
min_idx = np.argmin(min_idx)
min_x = theta[min_idx]
params_min = _params[min_idx]
print(params_min)
##Draw 20000 samples
leakage = posterior.set_default_x(x_o)
posterior_samples = posterior.sample((1000,), x=x_o) #, sample_with_mcmc=True
log_prob = posterior.log_prob(posterior_samples, x=x_o).numpy() 
params = posterior_samples.numpy()[np.argmax(log_prob)] #Take the sample with the highest log prob

#%% Plots
print(params)
## Plot the param / probability relationship
_ = utils.pairplot(posterior_samples, 
                    fig_size=(15,15), labels=["Cm", "taum", "El", "A", "b", "tauw", "DeltaT", "Vt", "VR", "refrac"])
N=1

plt.figure(20)
fi = generate_fi_curve(torch.tensor(params), save=False).numpy()
plt.plot(fi[0,:], label="simulated data")
neuron_data = x_o.numpy()
plt.plot(neuron_data, label="real data")
plt.plot(min_x, label="Found min")
plt.legend()

plt.figure(6)

for x in [0, 3, 5]:
    voltage, _ = adex_model(*params, realC_i=realC[x,:], record_v=True)
    plt.plot(realY[x,:], c='k', label="Real Sweep")
    plt.plot(voltage[0].v/mV, c="r", label="Post Min")
    voltage2, _ = adex_model(*params_min, realC_i=realC[x,:], record_v=True)
    plt.plot(voltage2[0].v/mV, c="g", label="Random Search Min")
plt.legend()
plt.show()

print("== Fitting Trace ==")
##TODO come up with other features to compute against reference trace
# %%
