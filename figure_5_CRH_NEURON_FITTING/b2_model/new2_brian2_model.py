from utils import *
from brian2 import *
from scipy import stats
from ipfx import feature_extractor
from ipfx import subthresh_features as subt


class brian2_model(object):

    def build_params_from_data(self):
        self.activeSweep = self.subthresholdSweep
        self.dt = compute_dt(self.realX)
        self.EL = compute_rmp(self.realY, self.realC)
        self.VT = compute_threshold(self.realX, self.realY, self.realC, self.activeSweep)
        self.spike_times = self._detect_real_spikes()
        self._compute_real_fi()
        #self._compute_subthreshold_features(self.realX[0], self.realY[0], self.realC[0])

    def _detect_real_spikes(self):
        return detect_spike_times(self.realX, self.realY, self.realC)

    def add_real_data(self,realX, realY, realC, spike_times, subthresholdsweep=[0], spikesweep=[5]):
        self.realX = realX
        self.realY = realY
        self.realC = realC
        self.spike_times = spike_times ##Passed in spike_times
        self.subthresholdSweep = subthresholdsweep ##If the user passed in the subthreshold sweep they want to fit to
        self.spikeSweep = spikesweep
        ##If the user passed in the spiking sweep to fit to
        return
    
    def _compute_real_fi(self):
        self.realFI, self.realISI = compute_FI_curve(self.spike_times, self._run_time)
        return

    def opt_spikes(self, param_dict, use_stored_spikes=False):
        self.__dict__.update(param_dict)
        error_s = np.zeros(self.N)
        
        for sweep in np.asarray(self.spikeSweep):
            self.activeSweep = sweep ##set the active sweep to the spiking sweep set by user
            spikes, _ = self.run_current_sim() ##Run the sim and grab the spikes
            self.temp_spike_array = spikes.spike_trains()
            sweep_error_s = []
            for unit in np.arange(0, self.N):
                temp = self.temp_spike_array[unit] / second
                if len(temp)< 1: 
                    temp_dist = 99999
                else:
                    try:
                        temp_dist = compute_spike_dist_euc(np.hstack((np.log10(temp[0]*1000), np.diff(temp)*1000)), np.hstack((np.log10(self.spike_times[self.activeSweep][0]*1000), np.diff(self.spike_times[self.activeSweep])*1000)))
                    except:
                        temp_dist = 99999
                sweep_error_s = np.hstack((sweep_error_s, temp_dist))
            error_s = np.vstack((error_s, sweep_error_s))
        error_s = np.sum(error_s, axis=0)
        del self.temp_spike_array
        return error_s

    def opt_trace(self, param_dict, during_stim=True):
        self.__dict__.update(param_dict)
        error_t = np.zeros(self.N)
        for sweep in np.asarray(self.subthresholdSweep):
            stim_t = find_stim_changes(self.realC[0,:])
            self.activeSweep = sweep
            _, traces = self.run_current_sim()
            if during_stim:
                sweep_error_t = np.apply_along_axis(compute_sse,1,(traces.v /mV)[:,stim_t[0]:stim_t[1]],self.realY[sweep,stim_t[0]:stim_t[1]])
            else:
                sweep_error_t = np.apply_along_axis(compute_sse,1,(traces.v /mV),self.realY[sweep,:])
            error_t = error_t + sweep_error_t
        return error_t

    def opt_trace_and_spike_mse(self, param_dict):
        return self.opt_full_mse(param_dict)

    def opt_full_mse(self, param_dict):
        self.__dict__.update(param_dict) ##Apply the passed in params to the model
        error_t = self.opt_trace(param_dict)
        error_fi = self.opt_FI(param_dict)
        error_s = self.opt_spikes(param_dict)
        
        return stats.gmean([error_t, error_s, error_fi]), error_t, error_fi, error_s

    def opt_ipfx_features(self,param_dict):
        if param_dict is not None:
            self.__dict__.update(param_dict)
        #First compute Features for subthreshold
        for sweep in np.asarray(self.subthresholdSweep):
            self.activeSweep = sweep
            _, traces = self.run_current_sim()


    def opt_FI(self,param_dict=None):
        if param_dict is not None:
            self.__dict__.update(param_dict)
        unit_wise_FI, unit_wise_isi = self.build_FI_curve()
        real_FI = self.realFI
        log_sim = np.nan_to_num(np.log10(unit_wise_FI), posinf=0, neginf=0)
        log_real = np.nan_to_num(np.log10(real_FI), posinf=0, neginf=0)
        unit_wise_isi_e = np.apply_along_axis(compute_mse,1,np.nan_to_num(np.log10(unit_wise_isi), posinf=0, neginf=0),np.nan_to_num(np.log10(self.realISI), posinf=0, neginf=0))/10
        unit_wise_error = np.apply_along_axis(compute_mse,1,log_sim,log_real)
        return unit_wise_error +  unit_wise_isi_e

    def build_FI_curve(self, param_dict=None):
        if param_dict is not None:
            self.__dict__.update(param_dict)
        spikes_full = []
        isi_full = []
        for sweep in np.arange(self.realX.shape[0]):
            self.activeSweep = sweep ##set the active sweep to the spiking sweep set by user
            spikes, _ = self.run_current_sim() ##Run the sim and grab the spikes
            self.temp_spike_array = spikes.spike_trains()
            neuron_wise_spikes = []
            neuron_wise_isi = []
            for p in np.arange(self.N):
                pspikes = self.temp_spike_array[p]
                if len(pspikes) > 0:
                    neuron_wise_spikes.append(len(pspikes))
                    spike_s = pspikes/ms
                    if len(spike_s) > 1:
                        neuron_wise_isi.append(compute_distro_mode(np.diff(spike_s)))
                    else:
                        neuron_wise_isi.append(0)
                else:
                    neuron_wise_spikes.append(0)
                    neuron_wise_isi.append(0)
            isi_full.append(np.nan_to_num(np.hstack((neuron_wise_isi))))
            spikes_full.append(np.hstack(neuron_wise_spikes))

        spikes_return = np.vstack(spikes_full)
        isi_return = np.vstack(isi_full)
        del self.temp_spike_array
        return spikes_return.T / self._run_time, isi_return.T /10

    def _internal_spike_error(self, x):
          temperror = compute_spike_dist(np.asarray(self.temp_spike_array[x] / second), self.spike_times[self.activeSweep]) 
          return temperror
    
    def _modified_mse(self, y, yhat):
        y_min = np.amin(y)
        y_max = np.amax(y)
        low_thres =(1.5 * np.amin(yhat))
        high_thres = np.amax(yhat) + (-0.5 * np.amax(yhat))
        if (np.amin(y) < low_thres) or (np.amax(y) > high_thres):
             return 99999
        else:
            return compute_mse(y, yhat)

    def _compute_subthreshold_features(self, x, y, c):
        stim_int = find_stim_changes(c) #from utils.py
        sweep_rmp = self._compute_rmp(y, c) #from utils.py
        sag = subt.sag(x, y, c, x[stim_int[0]], x[stim_int[-1]])
        print(sag)
   
    def _compute_rmp(self, dataY, dataC):
        deflection = np.nonzero(dataC)[0][0] - 1
        rmp1 = np.mean(dataY[:deflection])
        return rmp1

    def set_params(self, param_dict):
        self.__dict__.update(param_dict)

    def get_params(self):
        return self.__dict__

    def run_current_sweep(self, sweepNumber, param_dict=None):
        if param_dict is not None:
            self.__dict__.update(param_dict)
        self.activeSweep = sweepNumber
        return self.run_current_sim()
    
    def latent_to_fi(self, inputs):
      latent_npy = inputs.numpy() * 100
      param_latent = {"VT": latent_npy[:,0], "VR": latent_npy[:, 1], "a": latent_npy[:,2], "b": latent_npy[:,3], "tauw": latent_npy[:, 4], "DeltaT":latent_npy[:,5]}
      fi_curve, isi_curve = self.build_FI_curve(param_dict=param_latent)
      fi_curve = fi_curve /.7
      return_data = np.hstack((fi_curve, isi_curve))
      return tf.convert_to_tensor(return_data, dtype="float32")


