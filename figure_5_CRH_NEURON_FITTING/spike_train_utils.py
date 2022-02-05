import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from brian2 import *
from scipy import spatial

#% MISC utils %#
#Tools for loading or transforming spike trains

def build_n_train(isi, n=2):
    isi_corr = isi
    for p in np.arange(1,n):
         temp = np.hstack((isi, np.full(p, np.nan)))[p:]
         isi_corr = np.vstack((isi_corr, temp))
    
    return isi_corr.T[:-p, :]

def build_kde(isi_corr):
    isi_corr = np.log10(isi_corr)[:-1]
    unit_kde = stats.gaussian_kde(isi_corr.T)

    return unit_kde

def build_isi_from_spike_train(M, low_cut=0., high_cut=None, indiv=False):
    spike_trains = M.spike_trains() 
    ISI_ = []
    N = len(spike_trains.keys())
    if high_cut is None:
        for u in np.arange(N).astype(np.int):
            spikes = spike_trains[u] / second
            spikes_filtered = spikes[spikes>low_cut]
            ISI_.append( np.diff(spikes_filtered)*1000)
    else:
        for u in np.arange(N).astype(np.int):
            spikes = spike_trains[u] / second
            spikes_filtered = spikes[np.logical_and(spikes>low_cut, spikes<high_cut)]
            ISI_.append(np.diff(spikes_filtered)*1000)
    
    if indiv==False:
        ISI_ = np.hstack(ISI_)
        if ISI_.shape[0] < 1:
            ISI_ = np.array([3000,3000])
    return ISI_

def build_isi_from_spike_train_sep(M):
    spike_trains = M.spike_trains() 
    ISI_ = []
    N = len(spike_trains.keys())

    for u in np.arange(N).astype(np.int):
        ISI_ = np.hstack((ISI_, np.diff(spike_trains[u] / second)))
    ISI_ *= 1000
    if ISI_.shape[0] < 1:
        ISI_ = np.array([3000,3000])
    return ISI_

def intra_burst_hist(isi, low_cut=None):
    train_ = build_n_train(isi)
    less_ = train_[:,1]<=6
    bins = np.logspace(0,4)
    pre_burst = train_[less_, 0]
    if low_cut is not None:
        pre_burst = pre_burst[pre_burst>=low_cut]
    
    intra_burst = np.histogram(pre_burst, bins)[0]
    intra_burst = intra_burst / intra_burst.sum()
    return intra_burst, pre_burst

def tonic_hist(isi):
    train_ = build_n_train(isi)
    less_ = np.logical_and(train_[:,0]>=6,train_[:,1]>=6)
    bins = np.logspace(0,4)
    pre_burst = train_[less_, 0]
    intra_burst = np.histogram(pre_burst, bins)[0]
    intra_burst = intra_burst / intra_burst.sum()
    return intra_burst, pre_burst

def intra_burst_mean(isi, low_cut=10):
    intra_burst, pre_burst = intra_burst_hist(isi, low_cut=low_cut)
    mean = np.nanmean(pre_burst)
    median = np.nanmedian(pre_burst)
    return mean, median

def binary_spike_train(TIME, bin_size=5, end=None, binary=True):
    '''Converts Time array into binary spike train
    all units should be in ms'''
    if end is None:
        end = TIME[-1] + bin_size
    bins = np.arange(0, end, bin_size)
    hist,_ = np.histogram(TIME, bins)
    if binary:
        hist[hist>1] = 1
    return hist


def binned_hz_pop(isi, time, popsize=500, binsize=500):
    binned_isi_hz = []
    bins = np.arange(0, time, binsize)
    bins_right = np.arange(binsize, time + binsize, binsize)
    for x, x_r in zip(bins, bins_right):
        temp_isi_array = []
        for u in np.arange(popsize):
            temp_isi = isi[u] / second
            temp_isi *= 1000
            filtered_isi = temp_isi[np.logical_and(temp_isi>=x, temp_isi<x_r)]
            temp_isi_array.append((len(filtered_isi)/(binsize/1000)))
        binned_isi_hz.append(np.nanmean(temp_isi_array))
    return np.hstack(binned_isi_hz), bins

def equal_ar_size_from_list(isi_list):
    lsize = len(max(isi_list, key=len))
    new_list = []
    for x in isi_list:
        new_list.append(np.hstack((x, np.full((lsize-len(x)), np.nan))))
    return np.vstack(new_list)

def save_spike_train(isi, n=500, rand_num=30, filename='spike_trains.csv'):
    units_index = np.random.randint(0,n, rand_num)
    isi_trains = []
    for x in units_index:
        train = isi[x] / second
        isi_trains.append(train)
    isi_out = equal_ar_size_from_list(isi_trains)
    np.savetxt(f"{filename}", isi_out.T, fmt='%.18f', delimiter=',')

def save_spike_train_col(isi, filename='spike_trains_col.csv'):
    train = isi.i
    time = isi.t
    isi_out = np.vstack((train, time))
    np.savetxt(f"{filename}", isi_out.T, fmt='%.18f', delimiter=',')

    
    

def filter_bursts(isi, sil=25):
    n_train = build_n_train(isi, 2)
    silences_ind = np.where(np.logical_and(n_train[:,0]>=sil, n_train[:,1]<=6))[0]
    labeled_isi = np.zeros(isi.shape[0])
    bursts = []
    for x in silences_ind:
        temp_isi = isi[int(x)+1:]
        end_x = x+1
        for i in temp_isi:
            if i >= 10:
                break
            else:
                end_x += 1
        labeled_isi[int(x)+1:end_x]=2
        bursts.append(isi[int(x):end_x])
    labeled_isi[silences_ind+1] = 1
    non_burst = isi[labeled_isi==0]
    return labeled_isi,bursts, non_burst


def spikes_per_burst(isi):
    labels, burst, non_burst = filter_bursts(ISI_)
    lens = []
    for x in burst:
        lens.append(len(x)+1)
    bins = np.arange(2,max(lens))
    hist = np.histogram(lens, bins)
    return lens, hist, bins


#% Plotting Functions %#
# Functions to plot isi trains in various manners

def plot_xy(isiarray, lwl_bnd=0, up_bnd=4, fignum=2, color='k', alpha=0.05):
    plt.figure(fignum, figsize=(10,10))
    if isiarray.shape[0] > 0:
        plt.scatter(isiarray[:,0],isiarray[:,1], marker='o', alpha=alpha, color=color)
    plt.ylim( (pow(10,lwl_bnd), pow(10,up_bnd)) )
    plt.xlim( (pow(10,lwl_bnd), pow(10,up_bnd)) )
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('post isi (ms)')
    plt.xlabel('pre isi (ms)')

def plot_isi_hist(ISI, fignum=99):
    log_bins = np.logspace(0,4)
    plt.figure(fignum)
    plt.clf()
    plt.hist(ISI, log_bins)
    xscale('log')
    title("Network isi Dist")
    
def plot_intra_burst(isi, fignum=100, low_cut=10):
    intra_burst, pre_burst = intra_burst_hist(isi, low_cut=low_cut)
    log_bins = np.logspace(0,4)
    plt.figure(fignum)
    plt.clf()
    plt.hist(pre_burst, log_bins)
    xscale('log')
    title("Pre Burst Interval")
    return intra_burst

def plot_patterns(isi, rand_num, n=500, figsize=(25,5), colour=True, random=True):
    plt.figure(7,  figsize=figsize )
    plt.clf()
    if random:
        units_index = np.random.randint(0,n, rand_num)
    else:
        if n>1:
            units_index = np.arange(0, rand_num)
        else:
            units_index = [rand_num]
    for x in units_index:
        u_isi = isi[x] / second
        if len(u_isi) > 0:
            time =  u_isi
            u_isi = (np.diff(u_isi))*1000
            color = ['red', 'blue', 'green', 'purple', 'orange']
            plt.ylim( (1, pow(10,3.5)))
            if colour:
                plt.scatter(time, np.hstack((0,u_isi)), alpha=0.5)
            else:
                plt.scatter(time, np.hstack((0,u_isi)), alpha=0.5, c='grey')
            plt.yscale('log')
            plt.title(f'Unit {x}')

def plot_inst_freq(ISI, figsize=(25,5), colour=None):
    plt.figure(7,  figsize=figsize )
    plt.clf()
    time =  np.cumsum(ISI)
    freq_ = 1/ISI
    print(time.shape)
    print(freq_)
    if colour is not None:
        plt.scatter(np.hstack((0,time))[:-1], freq_, alpha=0.5, c=colour)
    else:
        plt.scatter(np.hstack((0,time))[:-1], freq_, alpha=0.5, c='k')
    plt.yscale('log')   
    
def plot_switching_units(isi, switch, num, burst_thres=0.2, tonic_thres=0.1, n=500, figsize=(25,5), colour=True):
    units_index = np.arange(n)
    good_units = []
    for x in units_index:
        u_isi = isi[x] / second
        if len(u_isi) > 0:
            start_isi = np.diff(u_isi[u_isi<switch]) * 1000
            end_isi = np.diff(u_isi[u_isi>=switch]) * 1000
            if len(start_isi)>0 and len(end_isi)>0:
                bursting = (100>enforce_burst(start_isi, burst_thres))
                no_burst = (100>enforce_no_burst(end_isi, tonic_thres))
                if np.logical_and(bursting, no_burst):
                    good_units.append(x)
    print(len(good_units))
    if len(good_units) >= num:
        units_to_use = np.random.choice(good_units, num, replace=False)
        for x in units_to_use:
                plt.figure(x,  figsize=figsize )
                plt.clf()
                u_isi = isi[x] / second
                time =  u_isi
                u_isi = 1/(np.diff(u_isi))
                color = ['red', 'blue', 'green', 'purple', 'orange']
                plt.ylim( (pow(10,-1), pow(10,3)) )
                plt.scatter(time, np.hstack((0,u_isi)), alpha=0.7, c='k')
                plt.yscale('log')
                plt.title(f'Unit {x}')
    else:
        print("not enough units passing the threshold")
    return len(good_units)
            
#% Distance functions %#
#Various Functions (and internal functions) to measure distances between spike trains
#mostly focused on time-indepedent distances

def kde_dist(kde1, kde2):
    dist = kde1[0].integrate_kde(kde2[0])
    norm2 = kde2[0].integrate_kde(kde2[0])
    norm1 = kde1[0].integrate_kde(kde1[0])
    dist = 1/(dist/ (norm1+norm2))
    return dist

def pdist_kde(list_kde):
    ar_kde = np.vstack(list_kde)
    cd_mat = spatial.distance.pdist(ar_kde, kde_dist)
    return cd_mat

def isi_kde_dist(isi1, isi2):
    #turn to 2d train
    isi_corr1, isi_corr2 = build_n_train(isi1), build_n_train(isi2)
    #build kdes
    kde1, kde2 = build_kde(isi_corr1), build_kde(isi_corr2)

    dist = kde_dist([kde1], [kde2])

    return dist

def ks_isi(isi1, isi2):
    dist = stats.ks_2samp(isi1, isi2)[0]
    return dist

def intra_burst_dist(isi1, isi2, low_cut=10, plot=False):
    intra_burst1, pre_burst1 = intra_burst_hist(isi1, low_cut=low_cut)
    intra_burst2, pre_burst2 = intra_burst_hist(isi2, low_cut=low_cut)
    bins = np.linspace(0,4)
    hisi1 = np.histogram(np.log10(pre_burst1), bins)[0].astype(np.float64)
    hisi2 = np.histogram(np.log10(pre_burst2), bins)[0].astype(np.float64)
    if hisi1.sum() < 1e-5:
        hisi1 += 0.5
    if hisi2.sum() < 1e-5:
        hisi2 += 0.5
    hisi1 /= hisi1.sum()
    hisi2 /= hisi2.sum()
    dist = stats.wasserstein_distance(np.arange(hisi1.shape[0]), np.arange(hisi2.shape[0]), hisi1, hisi2)
    
    if plot:
        plt.figure(565)
        plt.clf()
        bins = np.linspace(0,4)
        plt.hist(np.log10(pre_burst1), bins, alpha=0.25, density=True)
        plt.hist(np.log10(pre_burst2), bins, alpha=0.25, density=True)
        
    return dist

def tonic_dist(isi1, isi2, plot=False):
    intra_burst1, pre_burst1 = tonic_hist(isi1)
    intra_burst2, pre_burst2 = tonic_hist(isi2)
    bins = np.linspace(0,4)
    hisi1 = np.histogram(np.log10(pre_burst1), bins)[0].astype(np.float64)
    hisi2 = np.histogram(np.log10(pre_burst2), bins)[0].astype(np.float64)
    if hisi1.sum() < 1e-5:
        hisi1 += 0.5
    if hisi2.sum() < 1e-5:
        hisi2 += 0.5
    hisi1 /= hisi1.sum()
    hisi2 /= hisi2.sum()
    dist = stats.wasserstein_distance(np.arange(hisi1.shape[0]), np.arange(hisi2.shape[0]), hisi1, hisi2)
    
    if plot:
        plt.figure(565)
        plt.clf()
        bins = np.linspace(0,4)
        plt.hist(np.log10(pre_burst1), bins, alpha=0.25, density=True)
        plt.hist(np.log10(pre_burst2), bins, alpha=0.25, density=True)
        
    return dist

def enforce_no_burst(isi, threshold, print_index=False):
    burst_count = len(isi[isi<=8])
    overallsize = len(isi)
    burst_index = burst_count/overallsize
    if print_index:
        print(burst_index)
    if burst_index > threshold:
        return burst_index * 1000
    else:
        return burst_index

def enforce_burst(isi, threshold, print_index=False):
    burst_count = len(isi[isi<=8])
    overallsize = len(isi)
    burst_index = burst_count/overallsize
    if print_index:
        print(burst_index)
    if burst_index < threshold:
        return (1/(burst_index + 1e-4)) * 1000
    else:
        return (1/(burst_index + 1e-4))

def compute_burst_index(isi):
    burst_count = len(isi[isi<=8])
    overallsize = len(isi)
    burst_index = burst_count/overallsize
    return burst_index

def isi_hist(isi1):
    bins = np.logspace(0,4)
    hisi1 = np.histogram(isi1, bins, density=False)[0].astype(np.float64)
    hisi1 /= hisi1.sum()
    if np.all(hisi1==0):
        hisi1 += 1e-6
    return hisi1

def emd_isi(isi1,isi2):
    bins = np.logspace(0,4)
    hisi1 = np.histogram(isi1, bins, density=False)[0].astype(np.float64)
    hisi2 = np.histogram(isi2, bins, density=False)[0].astype(np.float64)
    hisi1 /= hisi1.sum()
    hisi2 /= hisi2.sum()
    hisi1 = np.nan_to_num(hisi1, nan=1e-6)
    hisi2 = np.nan_to_num(hisi2, nan=1e-6)
    if np.all(hisi1==0):
        hisi1 += 1e-6
    if np.all(hisi2==0):
        hisi2 += 1e-6
    
    hisi1 = np.nan_to_num(hisi1, nan=1e-6)


    dist = stats.wasserstein_distance(np.arange(hisi1.shape[0]), np.arange(hisi1.shape[0]), hisi1, hisi2)

    return dist


def sliced_wasserstein(X, Y, num_proj):
    '''Takes:
        X: 2d (or nd) histogram normalized to sum to one
        Y: 2d (or nd) histogram normalized to sum to one
        num_proj: Number of random projections to compute the mean over
        ---
        returns:
        mean_emd_dist'''
     #% Implementation of the (non-generalized) sliced wasserstein (EMD) for 2d distributions as described here: https://arxiv.org/abs/1902.00434 %#
    # X and Y should be a 2d histogram 
    # Code adapted from stackoverflow user: Dougal - https://stats.stackexchange.com/questions/404775/calculate-earth-movers-distance-for-two-grayscale-images
    dim = X.shape[1]
    ests = []
    for x in range(num_proj):
        # sample uniformly from the unit sphere
        dir = np.random.rand(dim)
        dir /= np.linalg.norm(dir)

        # project the data
        X_proj = X @ dir
        Y_proj = Y @ dir

        # compute 1d wasserstein
        ests.append(stats.wasserstein_distance(np.arange(dim), np.arange(dim), X_proj, Y_proj))
    return np.mean(ests)

def isi_swasserstein_2d(isi1, isi2, bin=28, log=True, plot=False, savefig_name=''):
    
    xbins = np.linspace(0,4,bin+1)
    ybins = np.linspace(0,4,bin+1)
    isi1, isi2 = np.log10(isi1), np.log10(isi2)
    isi_corr1, isi_corr2 = build_n_train(isi1), build_n_train(isi2)
    hist1, _, _ = np.histogram2d(isi_corr1[:,0], isi_corr1[:,1], bins=(xbins, ybins))
    hist2, _, _ = np.histogram2d(isi_corr2[:,0], isi_corr2[:,1], bins=(xbins, ybins))
    #print(hist1.max())
    #print(hist2.max())
    
    if np.all(hist1==0):
        hist1 += 0.5
    if np.all(hist2==0):
        hist2 += 0.5
    hist1 /= hist1.sum()
    hist2 /= hist2.sum()
    dist = sliced_wasserstein(hist1, hist2, 500)

    if plot:
        plt.figure(num=920)
        plt.clf(),
        plot_xy(10**isi_corr2, fignum=920, color='k', alpha=0.2)
        plot_xy(10**isi_corr1, fignum=920, color='r', alpha=0.2)
        
        #plt.tight_layout()
        plt.savefig(savefig_name)

    return dist

def biemd(isi1, isi2, lower=11, upper=200):
    #lower emd 
    lisi1 = isi1[isi1<=lower]
    lisi2 = isi2[isi2<=lower]
    lower_dist = emd_isi(lisi1, lisi2)
    #upper emd
    uisi1 = isi1[isi1>=upper]
    uisi2 = isi2[isi2>=upper]
    upper_dist = emd_isi(uisi1, uisi2)
    return lower_dist+upper_dist