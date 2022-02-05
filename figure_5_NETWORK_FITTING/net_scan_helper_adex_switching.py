import os
import sys
import time
from brian2 import *
from joblib import load as joblib_load
from utils import *
from spike_train_utils import *
from loadEXTRA import loadISI
import numpy as np
import os
prefs.codegen.target = 'cython'   # weave is not multiprocess-safe!
cache_dir = os.path.expanduser(f'~/.cython/brian-pid-{os.getpid()}')
prefs.codegen.runtime.cython.cache_dir = cache_dir

prefs.codegen.runtime.cython.multiprocess_safe = False
BrianLogger.suppress_hierarchy('brian2.codegen')
BrianLogger.suppress_hierarchy('brian2.groups.group.Group.resolve.resolution_conflict')

def network_scan(ee,ii, wcrh, de, di, dcrh, _vt, _vr, input_hz, p_ie, p_ei, i_pr, e_pr, b_change, adaptation_time_constant):
    start_scope()
    i = int(np.random.randint(9999, size=1) + time.time())
    print("=== Network scan started ===")
    start_time = time.time()
    # network parameters
    N = 1e3


    kde = joblib_load("sampled_cell.joblib")
    sample = kde
    columns=['Cm', 'Taum', 'EL', 'a', 'b', 'tauw', 'DeltaT', 'VT', 'VR']
    start_scope()
    start_time = time.time()
    seed(4323)
    # network parameters
    

    # neuron parameters
    pr = 1
    EL = -67.91702674 * mV
    VT = _vt * mV
    VR = _vr * mV
    LTS = {'C':sample[:,0]*pF, 'taum': sample[:, 1]*ms, 'gL': ((sample[:,0]*pF)/farad)/((sample[:, 1]*ms)/second) *siemens , 'tauw': sample[:,5]*ms, 
       'a': sample[:,3]*nS, 'b': sample[:,4]*pA,
      'DeltaT': sample[:, 0]*mV, 'refrac':sample[:, 9]*ms} # low-threshold spiking(?)
    FS = {'C': 21.5*pF, 'taum':26.45276208*ms, 'gL': 0.8182804*nS, 'tauw': 13.5686673*ms, 'a': 0.*nS, 'b': 3*pA, 'DeltaT': 12.003*mV }


    Ee = 0 * mvolt
    Ei = -80 * mvolt
    taue = de * msecond
    tauCRH = dcrh* msecond
    taun = 1 * msecond
    taui = di * msecond
    wCRH = wcrh * nS
    we = ee * nS
    wn = .000001 * nS
    wi = ii * nS
    taup = 400000000. *msecond
    taubr = 800000000000 * msecond


    #Some cells will be by default in single spiking
    #per = int(0.0 * 500)
    #rand_id = np.random.randint(0, 500, per)
    pr_ar = np.full(1000, 1.0)
    #pr_ar[rand_id] = np.clip(np.random.rand(per), 0.002, 0.6)
    b_ar = np.full(500, LTS['b']/pA)*pA
    #LTS['tauw'][rand_id] *= 2
    #b_ar[rand_id] = (18.14 * (5 + ((1-pr_ar[rand_id] + rand(per))* 20)))*pA#(2 + np.random.rand(per))*20)*pA
    #taup
    taup = np.full(1000, taup/second)*second
    taubr = np.full(1000, taubr/second)*second
    
    



    eqs = Equations('''
    dv/dt = ( gL*(EL-v) + gL*DeltaT*exp((v - VT)/DeltaT) + ge*(Ee-v) + gi*(Ei-v) - w) * (1./C) : volt (unless refractory)
    d_I = clip(( ge*(Ee-v) + gi*(Ei-v) - w), -5550*pA, 5550*pA) : amp 
    ge_I = ge*(Ee-v) : amp
    gi_I = gi*(Ei-v) : amp
    dw/dt = ( a*(v - EL) - w ) / tauw : amp 
    dgi/dt = -gi*(1./taui) : siemens
    dy/dt = -y*(1./taue) : siemens
    dbd/dt = -bd*(1./taubr): amp
    dge/dt = (y-ge)/taue : siemens
    dpr/dt = (1./taup) * pr * (1 - (pr/1)): 1
    p_w = clip(pr, 0.01,1): 1
    br = b + bd : amp
    tauw : second
    taue : second
    taui : second
    a : siemens
    I : amp
    b: amp
    Vcut: volt
    VT : volt
    refrac : second
    taup: second
    taubr : second
    input_1 : 1
    input_2 : 1
    input_3 : 1
    C : farad
    taum : second
    gL: siemens
    DeltaT: volt
    clamp_lim : siemens
    p_e : 1
    ''')


    # build network
    P = NeuronGroup( N, model=eqs, threshold='v>Vcut', reset='v=VR; w+=br', refractory='refrac')
    P.pr = 1
    P.VT = VT
    P.Vcut = 0 *mV
    P.taui = taui
    #print(P.Vcut)
    CRH = P[:int(floor(0.5*N))]; GABA = P[int(floor(0.5*N)):]
    CRH.tauw = LTS['tauw']; CRH.a = LTS['a']; CRH.b = LTS['b']; CRH.C =LTS['C']; CRH.taum = LTS['taum']; CRH.gL = LTS['gL']; CRH.DeltaT = LTS['DeltaT']
    GABA.tauw = FS['tauw']; GABA.a = FS['a']; GABA.b = FS['b']; GABA.C =FS['C']; GABA.taum = FS['taum']; GABA.gL = FS['gL']; GABA.DeltaT = FS['DeltaT']
    GABA.taue = tauCRH; CRH.taue = taue
    GABA.refrac = 1*ms
    CRH.refrac = LTS['refrac']
    # connect
    EI = Synapses( CRH, GABA, on_pre='y=clip(wCRH+y, 0*nS, clamp_lim)' ) #*(rand()<p_w)
    EI.connect( True, p=p_ei )
    IE = Synapses( GABA, CRH, on_pre='gi=clip((wi*int(rand()<p_w)) + gi, 0*nS, clamp_lim)' ) #*(rand()<p_w)
    IE.connect( True, p=p_ie )
    P.input_1 = 1
    # poisson input
    input = PoissonInput( CRH, 'ge', 1, input_hz*Hz, weight='we*int(rand()<p_e)')
    # init
    P.v =  EL
    P.p_e = 1
    P.ge = (randn(len(P)) * 2 + 5) * 0
    P.gi = ((randn(len(P)) * 2 + 5) + 0) * 0
    P.clamp_lim = 5*nS
    P.taup=taup
    P.taubr=taubr
    P.pr=pr_ar
    CRH.bd=b_ar
    run(5*second)
    P.clamp_lim = 90*nS
    
    

    # monitor
    M = SpikeMonitor( CRH )
    V = StateMonitor(CRH, ['v', 'ge_I'], [0], dt=1*ms)
    M2 = SpikeMonitor( GABA )
    

    


    CRH.I = 0 * pA
    run(50*second)
    CRH.I = 0 * pA
    P.pr=i_pr
    P.p_e = e_pr
    #we=10*nS
    CRH.bd=(CRH.b/pA *  b_change)*pA
    CRH.tauw = (CRH.tauw/ms *  adaptation_time_constant)*ms
    run(50 * second)


    #Load the data to fit to
    burst_ref = np.load("unit_3_burst_match1.npy")
    non_burst_ref = np.load("unit_3_nonburst_match.npy")

    ref_hist = isi_hist(burst_ref)
    sim_isi = np.nan_to_num(build_isi_from_spike_train(M), posinf=0, neginf=0)
    sim_isi_strt = np.nan_to_num(build_isi_from_spike_train(M, low_cut=5, high_cut=45.5, indiv=True), posinf=0, neginf=0)
    sim_isi_end = np.nan_to_num(build_isi_from_spike_train(M, low_cut=55, indiv=True), posinf=0, neginf=0)
    if sim_isi.shape[0] < 50:
        sim_isi = np.full(50, 0)
    #ISICORR = build_n_train(sim_isi,2)
    print(f"=== Network end in {(time.time() - start_time)/60} ===")
    np_strt = np.zeros(ref_hist.shape[0])
    np_end = np.zeros(ref_hist.shape[0])
    if np.all(sim_isi==0):
        dist = 999999
        mean, median = 0,0
        dist_units = 99999
        dist_strt = 99999
        dist_end = 99999
        mean_strt = [9999, 9999]
        mean_end = [9999, 9999]
        np_strt = np.zeros(ref_hist.shape[0])
        np_end = np.zeros(ref_hist.shape[0])
    else:
        dist = 0
        dist_strt = 0
        dist_end = 0
        mean_strt = []
        mean_end = []
        np_strt = np.zeros(ref_hist.shape[0])
        np_end = np.zeros(ref_hist.shape[0])
        for row in np.arange(len(sim_isi_strt)):
            unit_strt = sim_isi_strt[row]
            unit_end = sim_isi_end[row]
            if len(unit_strt) < 5 or len(unit_strt[unit_strt<2]) > 20:
                dist += 999
                mean_strt.append(99)
                mean_end.append(99)
                np_strt += np.zeros(ref_hist.shape[0])
                np_end += np.zeros(ref_hist.shape[0])
                continue
            elif len(unit_end) <5:
                dist += 99
                mean_strt.append(99)
                mean_end.append(99)
                np_strt += np.zeros(ref_hist.shape[0])
                np_end += np.zeros(ref_hist.shape[0])
                continue
            else:
                dist_strt = emd_isi(burst_ref, unit_strt)
                #dist_strt_middle = 1 / (emd_isi(non_burst_ref, unit_strt))
                #dist_strt += dist_strt_middle
                dist_end = emd_isi(non_burst_ref, unit_end)
                hisi_strt = isi_hist(unit_strt)
                hisi_end = isi_hist(unit_end)
                np_strt += hisi_strt
                np_end += hisi_end
                mean_strt.append(dist_strt)
                mean_end.append(dist_end)
                dist = dist_strt + dist #+ dist_end
    ## Take the top 250 distances as the dist.
    # 
    dist = np.nanmean(np.sort(mean_strt)[:100])  
    dist_e = np.nanmean(np.sort(mean_end)[:100])
    dist = dist + dist_e
    #plot emd dist 
    dist_arg = np.argsort(mean_strt)[:100]
    dist_temp = []
    dist_e_temp = []
    #for temp_arg in dist_arg:
       # unit_strt = sim_isi_strt[temp_arg]
       # unit_end = sim_isi_end[temp_arg]
       # dist_temp.append(isi_swasserstein_2d(unit_strt, burst_ref, plot=False, savefig_name=f'output//burst_{dist}_{i}_isi.png'))
      #  dist_e_temp.append(isi_swasserstein_2d(unit_end, non_burst_ref, plot=False, savefig_name=f'output//tonic_{dist}_{i}_isi.png'))
    #isi_swasserstein_2d(unit_strt, burst_ref, plot=True, savefig_name=f'output//burst_{dist}_{i}_isi.png')
    #isi_swasserstein_2d(unit_end, non_burst_ref, plot=True, savefig_name=f'output//tonic_{dist}_{i}_isi.png')
    #dist = np.nanmean(dist_temp)
    #dist_e = np.nanmean(dist_e_temp)
    #dist = dist + dist_e
    mean_volt = np.nanmean(V.v[0]/mV)
    
    #mean_volt_dist = np.abs(mean_volt - -68)/10
    #dist += mean_volt_dist
    mean_ge = np.median(np.clip(V.ge_I[0]/pA,0, 99999)[:50000]) 
    mean_ge_dist = np.abs((mean_ge-100))/10
    #dist = dist #+ (mean_ge_dist)
    #print(f"Mean volt is {mean_volt} with mean stim {mean_ge} and {mean_ge_dist}, and dist {dist}")
    #dist = dist + dist_e
    unit_wise_isis = []
    burst_rates = []
    low_state = []
    high_state = []
    firing_rate = []
    for u in np.arange(N//2).astype(np.int):
        ISI_ = sim_isi_strt[u]
        firing_rate.append(len(ISI_)/40)
        if len(ISI_) > 0:
            label_isi, bursts, non_bursts = filter_bursts(ISI_)
            
            burst_rate = (len(bursts) / 40)
            burst_rates.append(burst_rate)
            low_state.append(np.nanmean(non_bursts))
            #high_state.append(np.nanmean(bursts))
        else:
            burst_rates.append(0)
    burst_rates = np.hstack(burst_rates)
    no_burst_ = np.where(burst_rates<0.1)[0].shape[0]
    burst_unit = burst_rates.shape[0] - no_burst_

    burst_dist = np.abs(np.nanmean(burst_rates) - 0.18) * 10
    dist += burst_dist

    fr_dist = np.abs(np.nanmean(firing_rate) - 3.6) 
    dist += fr_dist
    perc = (burst_unit / burst_rates.shape[0]) * 100
    perc_dist = np.abs((perc-50))/10
    dist += perc_dist
    params = np.vstack([dist, np.nanmean(mean_strt), np.nanmean(mean_end), perc, np.nanmean(burst_rates), np.nanmean(firing_rate), ee,ii, wcrh, de, di, dcrh, _vt, _vr, input_hz])
    #params = np.hstack([dist, mean, median, ee, ii, bb, de, di])
    
    #pickle out the data
    np_strt /= 500
    np_end /= 500

    print(f"Mean volt is {mean_volt} mV with mean stim {mean_ge} pA, and dist {dist}, and pecent burst {perc}, and burst rate {np.nanmean(burst_rates)} and firing rate {np.nanmean(firing_rate)}")
    #np.save(f"C:\\Users\\SMest\\scan_fits_res\\{i}_res.npy", np.hstack((np_strt, np_end)))

    #np.savetxt(f"output//time_series//{i}_res.csv", np.hstack((params, spikes)).T,fmt='%.14f', delimiter=',')
    np.savetxt(f"C:\\Users\\SMest\\scan_fits_res\\{i}_res.csv", params.T ,fmt='%.14f', delimiter=',')
    #i+=1
    if True:
        figure(423, figsize=(25,5))
        
        clf()
        plot_patterns(M.spike_trains(), 10) # change 1 to 500 -> will plot 500 random neurons
        axhline((6))
        axhline((1))
        xlabel( 'Time (s)' ); ylabel( 'ISI' )
        title(f"Network Percent Burst {perc}")
        
        savefig(f"C:\\Users\\SMest\\scan_fits\\{dist}_{i}_1isi.png")
        figure(423, figsize=(5,5))
        plt.clf()
        subplot(2,1,1)
        plt.plot(V.ge_I[0]/pA)
        plt.xlim(500, 50000)
        plt.ylim(-300, 300)
        subplot(2,1,2)
        plt.bar(np.arange(np_strt.shape[0]),np_strt, alpha=0.2,)
        plt.bar(np.arange(np_strt.shape[0]),isi_hist(burst_ref), alpha=0.4, zorder=9999)
        savefig(f"C:\\Users\\SMest\\scan_fits\\volt\\{dist}_{i}_1volt.png")
    return np.nan_to_num(dist, nan=9999)
    
if __name__ == '__main__':
    freeze_support()