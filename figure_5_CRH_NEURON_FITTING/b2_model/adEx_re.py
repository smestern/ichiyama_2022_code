from utils import *
from brian2 import *
from b2_model.brian2_model import brian2_model
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects

def neuronwise_error(p, voltage, temp_spike_array, sweep):
                    pspikes = temp_spike_array[p] #Grab that neurons spike times
                    temp_isi = 0
                    temp_spikes = 0
                    if len(pspikes) > 0: #If there are any spikes
                        spikes_full[p].append(len(pspikes)) #Count the number of spikes and add it to the full array
                        spike_s = pspikes/ms #get the spike time in ms
                        if len(spike_s) > 1: #If there is more than one spike
                            temp_isi = np.nanmean(np.diff(spike_s)) #compute the mode ISI
                    ##Compute Subthresfeatures
                    temp_rmp = compute_rmp(voltage[p].v/mV.reshape(1,-1), sweep.reshape(1,-1)) #compute the beginning Resting membrane
                    temp_deflection = compute_steady_hyp(voltage[p].v/mV.reshape(1,-1), sweep.reshape(1,-1)) #compute the end
                    temp_subt = np.hstack((temp_rmp, temp_deflection))

                    #compute Sweepwisemin
                    temp_min = compute_min_stim(voltage[p].v/mV, voltage[0].t/second, strt=0.62, end=1.0)
                    
                    return temp_isi, temp_spikes, temp_subt, temp_min


class adExModel(brian2_model):
    ''' 
    Represents an adex model object that can be called while storing important params that don't need to change.
    To initialize, call the object. In addition data can be passed a dict in the format {'Cm': 19} etc.
    Real data to model can be passed in using add_real_data
    otherwise certain values can be infered from the data using build_params_from_data()
    For usage example see single_neuron_spike_and_trace_fitter_v2.py
    ____
    Takes:
    param_dict: as dictionary of paramters to apply to the model
    Returns:
    a default adexmodel
    '''
    
    
    def __init__(self, param_dict=None):
        ## Default Values
        self.N = 1
        self._run_time = 3
        self.Cm = 18.
        self.EL = -65.
        self.VT = -50.
        self.VR = self.EL
        self.taum = 200.
        self.tauw = 150.
        self.a = 0.01
        self.b = 5.
        self.refractory = 5
        self.refrac = 0
        self.DeltaT = 5
        self.dt = 0.01
        self.activeSweep = 0
        self.realX = np.full((4,4), np.nan)
        self.realY = np.full((4,4), np.nan)
        self.realC = np.full((4,4000), 0)
        self.spike_times = np.full((4,4), np.nan)
        self.subthresholdSweep = 0
        self.spikeSweep = 5
        self.bias = 0
        #passed params
        if param_dict is not None:
            self.__dict__.update(param_dict) #if the user passed in a param dictonary we can update the objects attributes using it.
                # This will overwrite the default values if the user passes in for example {'Cm:' 99} the cm value above will be overwritten


    def run_current_sim(self, sweepNumber=None, param_dict=None):
        
        if param_dict is not None:
            self.__dict__.update(param_dict)
        if sweepNumber is not None and sweepNumber != self.activeSweep:
            self.activeSweep = sweepNumber
        
        
        start_scope()
        
        temp_in = self.realC[self.activeSweep,:] + (self.bias)
        in_current = TimedArray(values = temp_in * pamp, dt=self.dt * ms)
        
        # Membrane Equation
        eqs = Equations('''
        dv/dt = ( gL*(EL-v) + gL*DeltaT*exp((v - VT)/DeltaT) + I - ((w * I_true)/pA) ) * (1./C) : volt (unless refractory)
        dw/dt = ( a*(v - EL) - w ) / tauw : amp
        tauw : second
        a : siemens
        b : amp
        C : farad
        taum : second
        gL : siemens
        EL : volt
        VT : volt
        VR : volt
        Vcut : volt
        DeltaT : volt
        refrac : second
        I = in_current(t) : amp
        I_true = (int(I!=0*pA)) * pA : amp
        ''')


 
        # build network
        P = NeuronGroup( self.N, model=eqs, threshold='v>Vcut', reset='v=VR; w+=b', method='euler', refractory='refrac')
        CRH = P; 
        CRH.tauw = self.tauw *ms; 
        CRH.b = self.b * pA; 
        CRH.a = self.a * nS; 
        CRH.C = self.Cm * pF; 
        CRH.taum = self.taum *ms;
        CRH.gL = (self.Cm * pF) / (self.taum * ms); 
        CRH.EL = self.EL* mV;
        CRH.VT = self.VT * mV;
        CRH.Vcut = 0 * mV;
        CRH.refrac = self.refrac*ms;
        CRH.DeltaT = self.DeltaT * mV; 
        
        CRH.VR = self.VR *mV
        

        # init
        P.v = self.EL * mV
        

        # monitor
        M = SpikeMonitor( CRH )
        V = StateMonitor( CRH, ["v"], record=True, dt=self.dt * ms)
        run(self._run_time * second)
        return M,V
    
    def adex_model(self, cm_i, taum_i, El_i, a_i, b_i, tau_a_i, Dt_i, Vt_i, VR_i, realC_i=None, record_v=False) -> [StateMonitor, SpikeMonitor]:
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
        in_current = TimedArray(values = (realC_i * pamp) +(self.bias * pamp), dt=self.dt * ms) 
        G1 = NeuronGroup(self.N, eqs, threshold='v > Vcut', reset='v = VR; w += b', method='Euler', refractory=1*ms)
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
        # record variables
        if record_v == True:
            #Only record voltage if explicity asked to save memory
            Mon_v = StateMonitor( G1, "v", record=True, dt=self.dt * ms)
        else:
            Mon_v = None
        Mon_spike = SpikeMonitor( G1 )
        run(self._run_time * second)
        return Mon_v, Mon_spike

    def generate_fi_curve(self, param_set):
        ''' Function to handle interfacing between SNPE and brian2, running each sweep and computing the FI curve.
        Takes:
        param_set (torch.tensor) of shape (num_units, params): A tensor containg the randomly sampled params, each row represents a single cell param set.

        Returns:
        vars (torch.tensor) of shape (num_units, measured vars): A tensor containing the FI curve, ISI-mode-Curve (most common ISI per Sweep), and subthreshold params
        '''
        param_list = param_set #Input is pytorch tensor converting it to a numpy array for brian2 purposes
        if param_list.shape[0] > 12:
            cm_i, taum_i, El_i, a_i, Dga_i, tau_a_i, Dt_i, Vt_i, VR_i = np.hsplit(param_list, 9) #Splits the variables out of the array
            ## Then flatten the variable arrays
            cm_i, taum_i, El_i, a_i, Dga_i, tau_a_i, Dt_i, Vt_i, VR_i = np.ravel(cm_i), np.ravel(taum_i), np.ravel(El_i), np.ravel(a_i), np.ravel(Dga_i), np.ravel(tau_a_i), np.ravel(Dt_i), np.ravel(Vt_i), np.ravel(VR_i)
        else:
            cm_i, taum_i, El_i, a_i, Dga_i, tau_a_i, Dt_i, Vt_i, VR_i = param_list
        spikes_full = [[] for x in np.arange(self.N)]
        isi_full = [[] for x in np.arange(self.N)]
        ##Generate an empty list of length N_UNITS which allows us to dump the subthreshold params into
        subthres_features = [[] for x in np.arange(self.N)]
        stim_min = [[] for x in np.arange(self.N)]
        for i, sweep in enumerate(self.realC): #For each sweep
            print(f"Simulating sweep {i}")
            voltage, spikes = self.adex_model(cm_i, taum_i, El_i, a_i, Dga_i, tau_a_i, Dt_i, Vt_i, VR_i, realC_i=sweep, record_v=True) #Run the adex model with the passed in params
            temp_spike_array = spikes.spike_trains() # Grab the spikes oriented by neuron in the network
            print("Simulation Finished")
            for p in np.arange(self.N): #For each neuron
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
                    temp_min = compute_min_stim(voltage[p].v/mV, voltage[0].t/second, strt=0.62, end=1.0)
                    stim_min[p].append(temp_min)
            
        neuron_avgs = np.vstack([np.nanmean(np.vstack(x), axis=0) for x in subthres_features]) #For each neuron, compute the mean accross sweeps
        #of the SubT features
        spikes_return = np.array(spikes_full) #Stack all the arrays together
        isi_return = np.array(isi_full) #Stack all the arrays together
        min_return = np.array(stim_min)
        return_full = np.hstack((spikes_return / 2, isi_return, neuron_avgs, min_return))
        return return_full

    def adex_model_refrac(self, cm_i, taum_i, El_i, a_i, b_i, tau_a_i, Dt_i, Vt_i, VR_i, refrac, realC_i=None, record_v=False) -> [StateMonitor, SpikeMonitor]:
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
        in_current = TimedArray(values = (realC_i * pamp) + (self.bias *pA), dt=self.dt * ms)
        G1 = NeuronGroup(self.N, eqs, threshold='v > Vcut', reset='v = VR; w += b', method='Euler', refractory='refrac')
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
        G1.Vcut = 0*mV
        G1.refrac = refrac*ms
        # record variables
        if record_v == True:
            #Only record voltage if explicity asked to save memory
            Mon_v = StateMonitor( G1, "v", record=True, dt=self.dt * ms)
        else:
            Mon_v = None
        Mon_spike = SpikeMonitor( G1 )
        run(self._run_time * second)
        return Mon_v, Mon_spike

    def generate_fi_curve_re(self, param_set):
        ''' Function to handle interfacing between SNPE and brian2, running each sweep and computing the FI curve.
        Takes:
        param_set (torch.tensor) of shape (num_units, params): A tensor containg the randomly sampled params, each row represents a single cell param set.

        Returns:
        vars (torch.tensor) of shape (num_units, measured vars): A tensor containing the FI curve, ISI-mode-Curve (most common ISI per Sweep), and subthreshold params
        '''
        param_list = param_set #Input is pytorch tensor converting it to a numpy array for brian2 purposes
        if param_list.shape[0] > 4:
            cm_i, taum_i, El_i, a_i, Dga_i, tau_a_i, Dt_i, Vt_i, VR_i, refrac = np.hsplit(np.vstack(param_list), 10) #Splits the variables out of the array
            ## Then flatten the variable arrays
            cm_i, taum_i, El_i, a_i, Dga_i, tau_a_i, Dt_i, Vt_i, VR_i, refrac = np.ravel(cm_i), np.ravel(taum_i), np.ravel(El_i), np.ravel(a_i), np.ravel(Dga_i), np.ravel(tau_a_i), np.ravel(Dt_i), np.ravel(Vt_i), np.ravel(VR_i), np.ravel(refrac)
        else:
            cm_i, taum_i, El_i, a_i, Dga_i, tau_a_i, Dt_i, Vt_i, VR_i, refrac = param_list
        spikes_full = [[] for x in np.arange(self.N)]
        isi_full = [[] for x in np.arange(self.N)]
        ##Generate an empty list of length N_UNITS which allows us to dump the subthreshold params into
        rmp_min = np.full((self.N, self.realC.shape[0]), 0)
        deflect_ar = np.full((self.N, self.realC.shape[0]), 0)
        stim_min = np.full((self.N, self.realC.shape[0]), 0)
        for i, sweep in enumerate(self.realC): #For each sweep
            print(f"Simulating sweep {i}")
            voltage, spikes = self.adex_model_refrac(cm_i, taum_i, El_i, a_i, Dga_i, tau_a_i, Dt_i, Vt_i, VR_i, refrac, realC_i=sweep, record_v=True) #Run the adex model with the passed in params
            temp_spike_array = spikes.spike_trains() # Grab the spikes oriented by neuron in the network
            voltage_ar = np.vstack([voltage[p].v/mV for p in np.arange(self.N)])
            voltage_t_ar = voltage[0].t/second
            print("Simulation Finished")
            neuronwise_stim_min = np.apply_along_axis(voltage_error_pass, 1, voltage_ar, voltage_t_ar, sweep, strt=0.62, end=1.0)
            #neuronwise_rmp_min = np.apply_along_axis(compute_rmp_passthrough, 1, voltage_ar, sweep)
            #neuronwise_deflection = np.apply_along_axis(compute_steady_hyp_passthrough, 1, voltage_ar, sweep)
            stim_min[:, i] = np.copy(neuronwise_stim_min[:, 0])
            rmp_min[:,i] = np.copy(neuronwise_stim_min[:, 1])
            deflect_ar[:, i] = np.copy(neuronwise_stim_min[:, 2])
            for p in np.arange(self.N): #For each neuron
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
            
        neuron_avgs = np.vstack((np.nanmean(rmp_min, axis=1), np.nanmean(deflect_ar, axis=1))).T #For each neuron, compute the mean accross sweeps
        #of the SubT features
        spikes_return = np.array(spikes_full) #Stack all the arrays together
        isi_return = np.array(isi_full) #Stack all the arrays together
        min_return = np.array(stim_min)
        return_full = np.hstack((spikes_return / 2, isi_return, neuron_avgs, min_return))
        return return_full


def voltage_error_pass(voltage_ar, voltage_t_ar, sweep, strt=0.62, end=1.0):
    stim = compute_min_stim(voltage_ar, voltage_t_ar, strt=strt, end=end)
    rmp = compute_rmp_passthrough(voltage_ar, sweep)
    deflect = compute_steady_hyp_passthrough(voltage_ar, sweep)
    return np.hstack((stim, rmp, deflect))

def compute_rmp_passthrough(ar_1, ar_2):
    return compute_rmp(ar_1.reshape(1,-1), ar_2.reshape(1,-1))

def compute_steady_hyp_passthrough(ar_1, ar_2):
    return compute_steady_hyp(ar_1.reshape(1,-1), ar_2.reshape(1,-1))

if __name__ == "__main__": 
    freeze_support()
