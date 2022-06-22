'''
Single neuron param fit
Rough coding space of several (optimizer, random, grid search) methods for optimizing
params
'''
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) #trying to log brian2 errors
import os
from brian2 import *
from loadNWB import *
from loadABF import *
from utils import *
import pandas as pd
from b2_model.adEx_re import adExModel
from b2_model.error import weightedErrorMetric
from b2_model.optimizer import snmOptimizer
from joblib import dump, load
import time
from scipy import stats
import multiprocessing
#to allow parallel processing
prefs.codegen.target = 'cython'  # weave is not multiprocess-safe!
cache_dir = os.path.expanduser(f'~/.cython/brian-pid-{os.getpid()}')
prefs.codegen.runtime.cython.cache_dir = cache_dir
prefs.codegen.runtime.cython.multiprocess_safe = False


#Global Settings
rounds = 5000
batch_size = 15000
## Grid search parameters
        #Ranges in lower -> upper
_Cm_range = (5., 25.) #in pF
_taum_range = (5., 30.1) #in ms
_EL_range = (-90, -60) #in ms
_VT_range = (-60., -20.) #in mV
_tauw_range = (0.01, 500.) #in ms
_a_range= (0.00005, 0.2) #in ns
_b_range = (0.0001, 210) #in pA
_DeltaT_range =(0.1,25.) #in mV
_DeltaA_range = (-10, 10) #in mV
_Ea_range = (-120.,-30.) #in mV
_VR_range = (-70., -30) #in mV
_Va_range = (-120, -20)
_refrac_range = (0, 2.)
vars = [_Cm_range, _taum_range, _EL_range, _a_range, _b_range, _tauw_range, _DeltaT_range , _VT_range, _VR_range,_refrac_range]
n_in_range = 4
#param grid
var_range_list = []
for v in vars:
     _range = np.linspace(v[0], v[1], n_in_range)
     var_range_list.append(_range)
var_range = tuple(var_range_list)
#param_dict = {'VT': var_range[0], 'tauw': var_range[1], 'a':var_range[2], 'b':var_range[3], 'DeltaT': var_range[4], 'VR': var_range[5], }
#gparam_dict = param_dict

_labels = ['Cm', 'taum', 'EL', 'a', 'b', 
                    'tauw','DeltaT', 'VT', 'VR', 
                    'refrac']


def run_optimizer(file, Cm, taum, optimizer='ng', rounds_=5000, batch_size_=15000):
    ''' Runs the optimizer for a given file, using user specfied params and rounds

    Takes:
    file (str) : A file path to a NWB file. To be fit to
    cm (float) : The precomputed cell capacitance
    taum (float) : The precomputed cell time constant
    (optional)
    optimizer (str) : the optimizer protocol to use, either 'ng' or 'snpe'
    rounds_ (int) : the number of rounds to run
    batch_size_ (int) : the number of neuron-param combos to try in parallel

    Returns:
    results : the best fit params for the cell
     '''
    global rounds
    global batch_size
    rounds=rounds_ #Set the global settings to the user passed in params
    batch_size=batch_size_
    model = load_data_and_model(file, Cm, taum) #load the nwb and model
    cell_id = file.split("\\")[-1].split(".")[0] #grab the cell id by cutting around the file path
    if optimizer == 'ng':
        results = nevergrad_opt(model, id=cell_id) #if the optimizer is ng pass in to ng
    if optimizer == 'snpe':
        results = SNPE_OPT(model, id=cell_id)
    return results

def SNPE_OPT(model, id='nan', use_post=True, refit_post=False, run_ng=True, early_stopping=True):
    ''' Samples from a SNPE posterior to guess the best possible params for the data. Optionally runs the NEVERGRAD differential
    evolution optimizer restricted to the SNPE-generated top 100 most probable parameter space following sample.
    takes
    ____
    model (a brian2_model object): with the cell data loaded into the objects properties
    id (str): a string containing the cell id. For saving the fit results
    use_post (bool): whether to use the prefit postierior (defaults to true)
    refit_post (bool): whether to refit the prefit postierior (defaults to false)
    run_ng (bool): whether to run a few rounds of the optimizer after sampling from SNPE (defaults to True)
    '''
    import torch
    from pickle import dump, load
    from sbi import utils as utils
    from sbi.inference.base import infer
    from sbi.inference import SNPE



    if use_post:
        #if the user requests we load the previously fit posterior
        with open("adexpost.pkl", "rb") as f:
            posterior = load(f)
        #also load the params used to fit those posteriors
        #theta = np.load("theta_ds.npy")
        #_params = np.load("params_ds.npy")
    else:
        print("TODO") ###Todo

    # Generate the X_o (observation) uses a combination of different params
    real_fi, real_isi = compute_FI_curve(model.spike_times, 2) #Compute the real FI curve
    #real_isi *= 10 #
    real_fi = np.hstack((real_fi, real_isi))
    real_rmp = compute_rmp(model.realY, model.realC)
    real_min = []
    real_subt = []
    for x in np.arange(model.realX.shape[0]):
        temp = compute_steady_hyp(model.realY[x, :].reshape(1,-1), model.realC[x, :].reshape(1,-1))
        temp_min = compute_min_stim(model.realY[x, :], model.realX[x,:], strt=0.62, end=1.2)
        real_subt.append(temp)
        real_min.append(temp_min)
    real_subt = np.nanmean(real_subt)
    x_o = torch.tensor(np.hstack((real_fi, real_rmp, real_subt, real_min))[:22], dtype=torch.float32)


    #set the default X, seems to speed up sampling
    leakage = posterior.set_default_x(x_o)
    posterior_samples = posterior.sample((10000,), x=x_o) #sample 100,000 points
    log_prob = posterior.log_prob(posterior_samples, x=x_o).numpy()  # get the log prop of these points
    params = posterior_samples.numpy()[np.argmax(log_prob)] #Take the sample with the highest log prob
    params = posterior.map(num_init_samples=10000, x=x_o).numpy()
    ##TODO maybe take the mode of the highest samples for each column?
    results_out = {'Cm': params[0], 'taum':params[1], 'EL': params[2], 'a': params[3], 'b': params[4],
                    'tauw': params[5], 'DeltaT': params[6], 'VT': params[7], 'VR': params[8], 'refrac': params[9]}
    min_dict = results_out
    plot_IF(min_dict, model)
    plt.savefig(f"output//{id}__checkpoint_fit_IF.png")
    plot_results(min_dict, model)
    plt.savefig(f"output//{id}__checkpoint_fit_vm.png")
    param_labels = list(results_out.keys())
    x_o = torch.tensor(np.hstack((real_fi, real_rmp, real_subt, real_min)), dtype=torch.float32)

    splits = [(0, 9),
           (9, 18),
            (18, 22),
         (22, 29)]
    splits = None
    weights = None
    errorCalc = weightedErrorMetric(x_o.numpy(), weights=weights, splits=splits)
    if run_ng:
        #Now run a few rounds of optimizer over the data
        import nevergrad as ng
        #top_100 = posterior_samples.numpy()[np.argsort(log_prob)[-100:]] #get the most probable 100 samples
        #high = np.apply_along_axis(np.amax,0, top_100) #take the high range of each column (each column is a different parameter)
        #low = np.apply_along_axis(np.amin,0, top_100) #take the low range
        var_pairs =  vars # np.transpose(np.vstack((low, high))) #stack them into low->pairs for each row
        var_list = []
        #constrain parameters to fit cm
        var_pairs[0] = (model.fitcm*0.5, model.fitcm*1.5)
        var_pairs[1] = (model.fittau*0.5, model.fittau*1.5)

        for i, var in enumerate(var_pairs):
            temp_var = ng.p.Scalar(lower=var[0], upper=var[1]) #Define them in the space that nevergrad wants
            print(f"{param_labels[i]}: low {var[0]} high {var[1]}")
            #temp_var.set_mutation(sigma=5)
            var_list.append(temp_var)
        var_tuple = tuple(var_list)

        var_dict = ng.p.Dict(Cm=var_tuple[0], taum=var_tuple[1], EL=var_tuple[2], a=var_tuple[3], b=var_tuple[4],  tauw=var_tuple[5],  DeltaT=var_tuple[6], VT=var_tuple[7], VR=var_tuple[8], refrac=ng.p.Scalar(lower=0, upper=2))
        model.set_params({'N': batch_size})
        budget = int(rounds * batch_size **3)
        opt = ng.optimizers.Portfolio(parametrization=var_dict, num_workers=batch_size, budget=budget)
        min_ar = []

        def conform_vt_vr(x):
            return x['VT'] > x['VR']


        opt.parametrization.register_cheap_constraint(conform_vt_vr)
        print(f"== Starting Optimizer with {rounds} rounds ===")
        for i in np.arange(rounds):
            print(f"iter {i} start")
            model.set_params({'N': batch_size, 'refractory':1})
            t_start = time.time()
            param_list_temp = []
            param_list = []
            param_list_temp = [opt.ask() for p in np.arange(batch_size)]
            param_list = [temp.value for temp in param_list_temp]
            param_list = pd.DataFrame(param_list)
            
            param_dict = {'Cm': param_list['Cm'].to_numpy(), 'taum': param_list['taum'].to_numpy(), 'EL': param_list['EL'].to_numpy(), 
                        'VT': param_list['VT'].to_numpy(), 'tauw': param_list['tauw'].to_numpy(), 'a':param_list['a'].to_numpy(), 
                        'b':param_list['b'].to_numpy(), 'DeltaT': param_list['DeltaT'].to_numpy(), 'VR': param_list['VR'].to_numpy()}
            print(f"sim {(time.time()-t_start)/60} min start")
            test = model.generate_fi_curve_re(param_list.to_numpy())
            #test =error_t, error_fi, error_s = model.opt_trace(param_dict), model.opt_FI(param_dict), model.opt_spikes(param_dict) /1000
            #error_t, error_fi, error_s = 0, 0 ,0
            if True:
                    if i == 0:
                        error = errorCalc.fit_transform(test)
                    else:
                        error = errorCalc.transform(test)
                        print(np.amin(error))
                    fi = test[:, :8]
                    isi = test[:, 8:18]
                    x_o_fi = x_o.numpy()[:8]
                    x_o_isi = x_o.numpy()[8:18]
                    subt = test[:, 18:22]
                    x_o_subt = x_o.numpy()[18:22]
                    stim_min = test[:, 22:]
                    x_o_stim_min = x_o.numpy()[22:]
                    y = np.apply_along_axis(compute_mse, 1, fi,  x_o_fi)
                    fi_er = np.copy(y) / 7
                    subt_er = np.apply_along_axis(compute_sse, 1, subt,  x_o_subt)
                    y += subt_er
                    stim_min_err = np.clip(np.apply_along_axis(compute_mse, 1, stim_min,  x_o_stim_min), 15, 99999) - 15
                    y += stim_min_err 
            #y = error_t + error_fi + error_s
            y = np.nan_to_num(y, nan=9999)
            print(f"sim {(time.time()-t_start)/60} min end")
            for k,row in enumerate(param_list_temp):
                opt.tell(row, y[k]) ##Tells the optimizer the param - error pairs so it can learn
            t_end = time.time()
            print(f"[CELL {id}] iter {i} excuted in {(t_end-t_start)/60} min, with error {np.amin(y)}")
            min_ar.append(np.sort(y)[:5])
            if (i>-1) and (i%1 == 0):
                #dump(opt, 'output//opt_checkpoint.joblib', compress=9)
                print('checkpoint saved')##Every 50 iter save the optimizer state (in case of crash) #
                results = opt.provide_recommendation().value 
                model.set_params({'N': 1})
                #returns a result containing the param - error matches
                min_dict = results
                results_out = results
                df = pd.DataFrame(results_out, index=[0])
                plot_IF(min_dict, model)
                plt.savefig(f"output//{id}_{i}_checkpoint_fit_IF.png")
                plot_results(min_dict, model)
                plt.savefig(f"output//{id}_{i}_checkpoint_fit_vm.png")
                df.to_csv(f'output//{id}_checkpoint_spike_fit_opt_CSV.csv')
                
            if early_stopping == True and len(min_ar)>1:
                check = _check_min_loss_gradient(min_ar, num_no_improvement=25)
                if check != True:
                    break
            #print(f"iter {i} excuted in {(t_end-t_start)/60} min, with error {np.amin(y)} with a min trace error {error_t[np.argmin(y)]} and FI error of {error_fi[np.argmin(y)]} and spike error of {error_s[np.argmin(y)]}") #    
        results = opt.provide_recommendation().value  #returns a result containing the param - error matches
        results_out = results
    min_dict = results_out
    plot_IF(min_dict, model)
    plt.savefig(f"output//{id}_checkpoint_fit_IF.png")
    plot_results(min_dict, model)
    plt.savefig(f"output//{id}_checkpoint_fit_vm.png")
    plt.pause(2)
    df = pd.DataFrame(results_out, index=[0])
    df['error_fi'] = fi_er[np.argmin(y)]
    df['error_subt'] = subt_er[np.argmin(y)]
    df.to_csv(f'output//{id}_spike_fit_opt_CSV.csv')
    return df


def nevergrad_opt(model, id='nan'):
    """Runs a nevergrad based optimizer on the model

    Args:
        model (b2_model): Model object containing a brian2 model and the linked intracellular data
        id (str, optional): Optional string to id the model. Defaults to 'nan'.

    Returns:
        DataFrame: a dataframe containing the best fit.
    """
    import nevergrad as ng

    #pack the variables into a dictionary for nevergrad
    var_list = []
    for var in vars:
        temp_var = ng.p.Scalar(lower=var[0], upper=var[1]) #splitting the highs and lows into two variables
        var_list.append(temp_var)
    var_tuple = tuple(var_list)
    var_dict = ng.p.Dict(Cm=var_tuple[0], taum=var_tuple[1], EL=var_tuple[2], a=var_tuple[3], b=var_tuple[4],  tauw=var_tuple[5],  DeltaT=var_tuple[6], VT=var_tuple[7], VR=var_tuple[8], refrac=var_tuple[9])
    #set the model parameters
    model.set_params({'N': batch_size})
    budget = int(rounds * batch_size)
    #create the optimizer
    opt = ng.optimizers.ParaPortfolio(parametrization=var_dict, num_workers=batch_size, budget=budget)
    
    print(f"== Starting Optimizer with {rounds} rounds ===")
    for i in np.arange(rounds):
        print(f"iter {i} start")
        #test the paramters
        model.set_params({'N': batch_size})
        t_start = time.time()
        #ask the optimizer for the parameters
        param_list_temp = []
        param_list = []
        for p in np.arange(batch_size):
            temp = opt.ask()
            param_list.append(temp.value)
            param_list_temp.append(temp)
        #pack the parameters into a dict
        param_list = pd.DataFrame(param_list)
        param_dict = {'Cm': param_list['Cm'].to_numpy(), 'taum': param_list['taum'].to_numpy(), 'EL': param_list['EL'].to_numpy(), 
                        'VT': param_list['VT'].to_numpy(), 'tauw': param_list['tauw'].to_numpy(), 'a':param_list['a'].to_numpy(), 
                        'b':param_list['b'].to_numpy(), 'DeltaT': param_list['DeltaT'].to_numpy(), 'VR': param_list['VR'].to_numpy(), 'refrac': param_list['refrac'].to_numpy()}
        print(f"sim {(time.time()-t_start)/60} min start")
        #test the parameters on the model
        error_t, error_fi, error_s = model.opt_trace(param_dict) / 10, model.opt_FI(param_dict), model.opt_spikes(param_dict) /1000
        error_fi = np.nan_to_num(error_fi, nan=999999) * 50
        error_t  = np.nan_to_num(error_t , nan=999999, posinf=99999, neginf=99999)
        y = error_fi + error_t + error_s
        y = np.nan_to_num(y, nan=999999)
        print(f"sim {(time.time()-t_start)/60} min end")
        for k,row in enumerate(param_list_temp):
            opt.tell(row, y[k]) ##Tells the optimizer the param - error pairs so it can learn
        t_end = time.time()
        print(f"iter {i} excuted in {(t_end-t_start)/60} min, with error {np.amin(y)} with a min trace error {error_t[np.argmin(y)]} and FI error of {error_fi[np.argmin(y)]} and spike error of {error_s[np.argmin(y)]}") #
        if (i>-1) and (i%1 == 0):
           
           print('checkpoint saved')
           results = opt.provide_recommendation().value 
           model.set_params({'N': 1})
           #returns a result containing the param - error matches
           min_dict = results
           results_out = results
           df = pd.DataFrame(results_out, index=[0])
           plot_IF(min_dict, model)
           plt.savefig(f"output//{id}_{i}_checkpoint_fit_IF.png")
           plot_results(min_dict, model)
           plt.savefig(f"output//{id}_{i}_checkpoint_fit_vm.png")
           df.to_csv(f'output//{id}_checkpoint_spike_fit_opt_CSV.csv')

    results = opt.provide_recommendation().value  #returns a result containing the param - error matches
    min_dict = results
    results_out = results
    plot_IF(min_dict, model)
    plot_results(min_dict, model)
    df = pd.DataFrame(results_out, index=[0])
    df.to_csv(f'output//{id}_spike_fit_opt_CSV.csv')
    return df
    

def plot_results(param_dict, model):
    realX, realY = model.realX, model.realY
    figure(figsize=(10,10), num=15)
    clf()
    model.set_params(param_dict)
    model.set_params({'N': 1})
    print(model.get_params())
    for x in [0,3,5]:
        spikes, traces = model.run_current_sweep(x)
        
        plot(realX[x,:], realY[x,:], label=f"Real Sweep {x}", c='k')
        plot(realX[x,:], traces.v[0] /mV, label="Sim sweep {x}", c='r')
        if len(spikes.t) > 0:
            scatter(spikes.t, np.full(spikes.t.shape[0], 60) ,label="Sim spike times", c='r')
    
    ylim(-100, 70)
    return

def plot_IF(param_dict, model):
    realX, realY, realC = model.realX, model.realY, model.realC
    figure(figsize=(10,10), num=13)
    clf()
    model.set_params(param_dict)
    model.set_params({'N':1})
    realspikes = model._detect_real_spikes()
    real_spike_curve,_ = compute_FI_curve(realspikes, model._run_time)
    
    simspikes,_ = model.build_FI_curve()
    simspikes = simspikes[0]
    mse = compute_mse(np.asarray(real_spike_curve),np.hstack(simspikes))
    plot(np.arange(simspikes.shape[0]), simspikes, label=f"Sim FI")
    plot(np.arange(simspikes.shape[0]), real_spike_curve, label=f"Real FI")
    plot(np.arange(simspikes.shape[0]), model.realFI, label=f"model Real FI")
    
    legend()

def load_data_and_model(file, Cm=11.70570803, taum=16.3721611190163, sweep_upper_cut=9):
    #% Opening ABF/Or nwb %#
    file_path = file
    extension = os.path.splitext(file_path)[1]
    if '.abf' in extension:
        realX, realY, realC = loadABF(file_path)
    else:
        realX, realY, realC,_ = loadNWB(file_path)
    index_3 = np.argmin(np.abs(realX[0,:]-2))
    realX, realY, realC = realX[:sweep_upper_cut,:index_3], realY[:sweep_upper_cut,:index_3], realC[:sweep_upper_cut,:index_3]
    sweeplim = np.arange(realX.shape[0])
    dt = compute_dt(realX)
    compute_el = compute_rmp(realY[:3,:], realC[:3,:])
    sweepwise_el = np.array([compute_rmp(realY[x,:].reshape(1,-1), realC[x,:].reshape(1,-1)) for x in np.arange(realX.shape[0])])
    sweep_offset = (sweepwise_el - compute_el).reshape(-1,1)
    realY = realY - sweep_offset


    spike_time = detect_spike_times(realX, realY, realC, sweeplim)
    thres = compute_threshold(realX, realY, realC, sweeplim)
    model = adExModel({'EL': compute_el, 'dt':dt, '_run_time':2, 'Cm': Cm, 'taum': taum, 'fitcm': Cm, 'fittau': taum})
    spike_sweeps = np.nonzero([len(x) for x in spike_time])[0]
    subt_sweeps = np.arange(spike_sweeps[0])
    model.add_real_data(realX, realY, realC, spike_time, subt_sweeps, spike_sweeps)
    model.build_params_from_data()
    return model

if __name__ == "__main__": 
    print("Running Script from commandline not yet supported")


# Functions below here I need to update or document or delete so just ignore for now

def _check_min_loss_gradient(min_ar, num_no_improvement=10):
    pass_check = True
    min_ar = np.array(min_ar)
    min_ar = np.nanmean(min_ar, axis=1)
    if min_ar.shape[0] <= num_no_improvement:
        pass
    else:
        x = np.arange(0, num_no_improvement)
        slope, _, _, p, _ = stats.linregress(x, min_ar[-num_no_improvement:])
        if (slope < 0.005 and slope > -0.005) and p < 0.01:
            pass_check = False
        print(slope)
    
    return pass_check