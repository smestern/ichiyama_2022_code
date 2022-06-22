import pandas as pd
from ipfx import feature_extractor
from scipy.stats import mode, pearsonr
from scipy import interpolate
from scipy.spatial import distance
from scipy.signal import find_peaks
import numpy as np
from brian2 import pF, pA, nS, mV, NeuronGroup, pamp, run, second, StateMonitor, ms, TimedArray, size, nan, array, reshape, \
    shape, volt, siemens, amp
from loadNWB import loadNWB
try:
    ### these are some libraries for spike train assesment not needed if you are not calling spike dist
    from elephant.spike_train_dissimilarity import victor_purpura_dist, van_rossum_dist
    from neo.core import SpikeTrain
    import quantities as pq 
except:
    print('Spike distance lib import failed')


from brian2 import plot


def detect_spike_times(dataX, dataY, dataC, sweeps=None, dvdt=7, swidth=10, speak=-10):
    # requires IPFX (allen institute).
    # works with abf and nwb
    swidth /= 1000
    if sweeps is None:
        sweepList = np.arange(dataX.shape[0])
    else:
        sweepList = np.asarray(sweeps)
    spikedect = feature_extractor.SpikeFeatureExtractor(filter=0, dv_cutoff=dvdt, max_interval=swidth, min_peak=speak)
    spike_list = []
    for sweep in sweepList:
        sweepX = dataX[sweep, :]
        sweepY = dataY[sweep, :]
        sweepC = dataC[sweep, :]
        try:
            spikes_in_sweep = spikedect.process(sweepX, sweepY, sweepC)  ##returns a dataframe
        except:
            spikes_in_sweep = pd.DataFrame()
        if spikes_in_sweep.empty == True:
            spike_list.append([])
        else:
            spike_ind = spikes_in_sweep['peak_t'].to_numpy()
            spike_list.append(spike_ind)
            
            
    return spike_list


def spikeIndices(V):
    # smooth the first difference of V; peaks should be clear and smooth
    dV = np.diff(V)
    dVsm, _, _, _ = smoothn(dV)
    
    # define spikes at indices where smoothed dV exceeds twice the standard deviation
    sigma = np.std(dVsm)
    spkIdxs, _ = find_peaks(dVsm, height=2*sigma)
    
    return spkIdxs


def compute_threshold(dataX, dataY, dataC, sweeps, dvdt=20):
    # requires IPFX (allen institute). Modified version by smestern
    # install using pip install git+https://github.com/smestern/ipfx.git Not on git yet
    # works with abf and nwb
    if sweeps is None:
        sweepList = np.arange(dataX.shape[0])
    else:
        sweepList = np.asarray(sweeps)
    spikedect = feature_extractor.SpikeFeatureExtractor(filter=0, dv_cutoff=dvdt)
    threshold_list = []
    for sweep in sweepList:
        sweepX = dataX[sweep, :]
        sweepY = dataY[sweep, :]
        sweepC = dataC[sweep, :]
        spikes_in_sweep = spikedect.process(sweepX, sweepY, sweepC)  ##returns a dataframe
        if spikes_in_sweep.empty == False:
            thres_V = spikes_in_sweep['threshold_v'].to_numpy()
            threshold_list.append(thres_V)

    return np.nanmean(threshold_list[0])


def compute_dt(dataX):
    dt = dataX[0, 1] - dataX[0, 0]
    dt = dt * 1000  # ms
    return dt

def compute_steady_hyp(dataY, dataC, ind=[0,1]):
    stim_index = find_stim_changes(dataC[0,:])
    mean_steady= np.nanmean(dataY[:, stim_index[ind[0]]:stim_index[ind[1]]])
    return mean_steady

def compute_rmp(dataY, dataC):
    deflection = np.nonzero(dataC[0, :])[0][0] - 1
    rmp1 = np.nanmean(dataY[:, :deflection])
    rmp2 = mode(dataY[:, :deflection], axis=None)[0][0]

    return rmp2

def find_stim_changes(dataI):
    diff_I = np.diff(dataI)
    infl = np.nonzero(diff_I)[0]
    
    '''
    dI = np.diff(np.hstack((0, dataI, 0))
    '''
    return infl


def find_downward(dataI):
    diff_I = np.diff(dataI)
    downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0]
    return downwardinfl

def exp_decay_1p(t, a, b1, alphaFast):
    return (a + b1*(1-np.exp(-t/alphaFast))) * 1000

def exp_decay_factor(dataT,dataV,dataI, time_aft=75):
     
        time_aft = time_aft / 100
        if time_aft > 1:
            time_aft = 1

        diff_I = np.diff(dataI)
        downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0]
        end_index = downwardinfl + int((np.argmax(diff_I)- downwardinfl) * time_aft)
        upperC = np.amax(dataV[downwardinfl:end_index])
        lowerC = np.amin(dataV[downwardinfl:end_index])
        diff = np.abs(upperC - lowerC)
        t1 = dataT[downwardinfl:end_index] - dataT[downwardinfl]
        SpanFast=(upperC-lowerC)*1*.01
        curve, pcov_1p = curve_fit(exp_decay_1p, t1, dataV[downwardinfl:end_index], maxfev=50000,  bounds=(-np.inf, np.inf))
        tau = 1/curve[2]
        return tau


def create_atf(data, filename="output.atf", rate=20000):
    """Save a stimulus waveform array as an ATF 1.0 file."""
    ATF_HEADER="""
            ATF	1.0
            8	2
            "AcquisitionMode=Episodic Stimulation"
            "Comment="
            "YTop=2000"
            "YBottom=-2000"
            "SyncTimeUnits=20"
            "SweepStartTimesMS=0.000"
            "SignalsExported=IN 0"
            "Signals="	"IN 0"
            "Time (s)"	"Trace #1"
            """.strip()
    out=ATF_HEADER
    for i,val in enumerate(data):
        out+="\n%.05f\t%.05f"%(i/rate,val)
    with open(filename,'w') as f:
        f.write(out)
        print("wrote",filename)
    return

def membrane_resistance(dataT,dataV,dataI):
    try:
        diff_I = np.diff(dataI)
        downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0]
        end_index = downwardinfl + int((np.argmax(diff_I)- downwardinfl)/2)
        
        upperC = np.mean(dataV[:downwardinfl-100])
        lowerC = np.mean(dataV[downwardinfl+100:end_index-100])
        diff = -1 * np.abs(upperC - lowerC)
        I_lower = dataI[downwardinfl+1]
        t1 = dataT[downwardinfl:end_index] - dataT[downwardinfl]
        #v = IR
        #r = v/I
        v_ = diff / 1000 # in mv -> V
        I_ = I_lower / 1000000000000 #in pA -> A
        r = v_/I_

        return r #in ohms
    except: 
        return np.nan


def plot_adex_state(adex_state_monitor):
    """
    Visualizes the state variables: w-t, v-t and phase-plane w-v
    from https://github.com/EPFL-LCN/neuronaldynamics-exercises/
    Args:
        adex_state_monitor (StateMonitor): States of "v" and "w"

    """
    import matplotlib.pyplot as plt
    plt.figure(num=12, figsize=(10,10))
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.plot(adex_state_monitor.t / ms, adex_state_monitor.v[0] / mV, lw=2)
    plt.xlabel("t [ms]")
    plt.ylabel("u [mV]")
    plt.title("Membrane potential")
    plt.subplot(2, 2, 2)
    plt.plot(adex_state_monitor.v[0] / mV, adex_state_monitor.w[0] / pA, lw=2)
    plt.xlabel("u [mV]")
    plt.ylabel("w [pAmp]")
    plt.title("Phase plane representation")
    plt.subplot(2, 2, 3)
    plt.plot(adex_state_monitor.t / ms, adex_state_monitor.w[0] / pA, lw=2)
    plt.xlabel("t [ms]")
    plt.ylabel("w [pAmp]")
    plt.title("Adaptation current")

def compute_sse(y, yhat):
    sse = np.sum(np.square(y - yhat))
    return sse


def compute_mse(y, yhat):
    mse = np.mean(np.square(y - yhat))
    return mse

def compute_se(y, yhat):
    se = np.square(y - yhat)
    return se


def equal_array_size_1d(array1, array2, method='append', append_val=0):
    ar1_size = array1.shape[0]
    ar2_size = array2.shape[0]
    if ar1_size == ar2_size:
        pass
    elif method == 'append':
        if ar1_size > ar2_size:
            array2 = np.hstack((array2, np.full(ar1_size - ar2_size, append_val)))
        elif ar2_size > ar1_size:
            array1 = np.hstack((array1, np.full(ar2_size - ar1_size, append_val)))
    elif method == 'trunc':
        if ar1_size > ar2_size:
            array1 = array1[:ar2_size]
        elif ar2_size > ar1_size:
            array2 = array2[:ar1_size]
    elif method == 'interp':
        if ar1_size > ar2_size:
            interp = interpolate.interp1d(np.linspace(1,ar2_size-1, ar2_size), array2, bounds_error=False, fill_value='extrapolate')
            new_x = np.linspace(ar2_size, ar1_size, (ar1_size - ar2_size))
            array2 = np.hstack((array2, interp(new_x)))
        elif ar2_size > ar1_size:
            interp = interpolate.interp1d(np.linspace(1,ar1_size-1, ar1_size), array1, bounds_error=False, fill_value='extrapolate')
            new_x = np.linspace(ar1_size, ar2_size, (ar2_size - ar1_size))
            array2 = np.hstack((array1, interp(new_x)))
    return array1, array2


def compute_spike_dist(y, yhat):
    '''
    Computes the distance between the two spike trains
    takes arrays of spike times in seconds
    '''
    #y, yhat = equal_array_size_1d(y, yhat, 'append')

    
    train1 = SpikeTrain(y*pq.s, t_stop=6*pq.s)
    train2 = SpikeTrain(yhat*pq.s, t_stop=6*pq.s)
    
    dist = victor_purpura_dist([train1, train2], q=0.20*pq.Hz)  
    ## Update later to compute spike distance using van rossum dist
    r_dist = dist[0,1] #returns squareform so just 
    return r_dist

def compute_spike_dist_euc(y, yhat):
    '''
    Computes the distance between the two spike trains
    takes arrays of spike times in seconds
    '''
    y, yhat = equal_array_size_1d(y, yhat, 'append', append_val=0)
    if len(y) < 1 and len(yhat) < 1:
        dist = 999
    else:
        dist = distance.euclidean(y, yhat)  
    
    r_dist = dist
    return r_dist


def compute_corr(y, yhat):

    y, yhat = equal_array_size_1d(y, yhat, 'append')
    y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)
    yhat = np.nan_to_num(yhat, nan=0, posinf=0, neginf=0)
    try:
        corr_coef = pearsonr(y, yhat)
    except:
        corr_coef = 0
    return np.amax(corr_coef)

def replace_nan(a):
    temp = a.copy()
    temp[np.isnan(a)] = np.nanmax(a)
    return temp

def drop_rand_rows(a, num):
    rows = a.shape[0]-1
    rows_to_drop = np.random.rarandint(0, rows, num)
    a = np.delete(a,rows_to_drop,axis=0)
    return a

def compute_distro_mode(x, bin=20, wrange=False):
    if wrange:
        bins = np.arange(np.amin(x)-bin, np.amax(x)+bin, bin)
    else:
        bins = np.arange(0, np.amax(x)+bin, bin)
    hist, bins = np.histogram(x, bins=bins)
    return bins[np.argmax(hist)]



def compute_corr_minus(y, yhat):

    y, yhat = equal_array_size_1d(y, yhat, 'append')
    y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)
    yhat = np.nan_to_num(yhat, nan=0, posinf=0, neginf=0)
    try:
        corr_coef = 1 - np.amax(pearsonr(y, yhat))
    except:
        corr_coef = 1
    return corr_coef

def compute_FI(spkind, dt, dataC):
    isi = [ dt*np.diff(x) for x in spkind ]
    f = [ np.reciprocal(x) for x in isi ]
    i = []
    for ii in range(len(dataC)):
        tmp = dataC[ii]
        tmp1 = spkind[ii][:-1]
        i.append(tmp[tmp1])
    return f, i, isi
    
def compute_min_stim(dataY, dataX, strt, end):
    #find the strt, end
    index_strt = np.argmin(np.abs(dataX - strt))
    index_end = np.argmin(np.abs(dataX - end))
    #Find the min
    amin = np.amin(dataY[index_strt:index_end])
    return amin

def compute_FI_curve(spike_times, time, bin=20):
    FI_full = []
    isi=[]
    for r in spike_times:
        if len(r) > 0:
            FI_full.append(len(r))
            if len(r) > 1:
                isi_row = np.diff(r)
                isi.append(np.nanmean(isi_row*1000))
            else:
                isi.append(0)
        else:
            FI_full.append(0)
            isi.append(0) 
    return (np.hstack(FI_full) /time), np.hstack(isi)
            
def add_spikes_to_voltage(spike_times,voltmonitor, peak=33, index=0):
    if len(spike_times) > 0:
            trace_round = np.around(voltmonitor.t/ms, decimals=0)
            spikes_round = np.around(spike_times, decimals=0)
            spike_idx = np.isin(trace_round, spikes_round)
            traces_v =  voltmonitor[index].v/mV
            traces_v[spike_idx] = peak
    else:
            traces_v =  voltmonitor[index].v/mV
    return traces_v
             


def LIF(file, C, g, sweepnum):
    dataX, dataY, dataC = loadNWB(file, return_obj=False)
    dt = compute_dt(dataX)
    dur = size(dataX, 1) * dt
    # E = compute_rmp(dataY, dataC) * mV
    E = np.mean(dataY[sweepnum, 0:2781]) * mV

    inpt = TimedArray(dataC[sweepnum, :] * pamp, dt=dt * ms)

    eqs = '''
    dv/dt = (-g*(v-E)+I)/C : volt
    I = inpt(t) : amp
    '''
    G = NeuronGroup(1, eqs, threshold='v>-45*mV', reset='v=-47*mV')
    M = StateMonitor(G, 'v', record=True)
    G.v = E

    run(dur * ms)
    t = M.t / ms / 1000
    v = M.v / mV

    vhat = array(dataY[sweepnum])
    vhat = reshape(vhat, shape(v))

    mse = compute_mse(v, vhat)

    return mse, t, v


def adIF(file, C, g, a, tauw, sweepnum):
    dataX, dataY, dataC = loadNWB(file, return_obj=False)
    dt = compute_dt(dataX)
    # dur = size(dataX, 1) * dt
    # E = compute_rmp(dataY, dataC) * mV
    E = np.mean(dataY[sweepnum, 0:2781]) * mV

    inpt = TimedArray(dataC[sweepnum, :] * pamp, dt=dt * ms)

    eqs = '''
    dv/dt = (-g*(v-E)+I-w)/C : volt
    dw/dt = (a*(v-E)-w)/tauw : amp
    I = inpt(t) : amp
    '''
    G = NeuronGroup(1, eqs)
    M = StateMonitor(G, ('v', 'w'), record=True)
    G.v = E

    run(5 * second)
    t = M.t / ms / 1000
    v = M.v / mV
    w = M.w

    return t, v, w
