from utils import *
from brian2 import *
from b2_model.brian2_model import brian2_model



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
        self.refractory = 1
        self.DeltaT = 5
        self.dt = 0.01
        self.activeSweep = 0
        self.realX = np.full((4,4), np.nan)
        self.realY = np.full((4,4), np.nan)
        self.realC = np.full((4,4000), 0)
        self.spike_times = np.full((4,4), np.nan)
        self.subthresholdSweep = 0
        self.spikeSweep = 5
        self.eqs = Equations('''
        dv/dt = ( gL*(EL-v) + gL*DeltaT*exp((v - VT)/DeltaT) + w*(Ea - v) + I ) * (1./C) : volt
        dw/dt = ( a/(1 + exp((Va - v)/DeltaA)) - w ) / tauw : siemens
        tauw : second
        a : siemens
        b : siemens
        C : farad
        taum : second
        gL : siemens
        EL : volt
        Ea : volt
        Va : volt
        VT : volt
        VR : volt
        Vcut : volt
        DeltaT : volt
        DeltaA : volt
        refrac : second
        I = in_current(t) : amp
        ''')
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
        
        temp_in = self.realC[self.activeSweep,:]
        in_current = TimedArray(values = temp_in * pamp, dt=self.dt * ms)
        
        # Membrane Equation
        eqs = self.eqs

        # build network
        P = NeuronGroup( self.N, model=eqs, threshold='v>Vcut', reset='v=VR; w+=b', method='euler', refractory=self.refractory*ms )
        CRH = P; 
        CRH.tauw = self.tauw *ms; 
        CRH.b = self.b * nS; 
        CRH.a = self.a * nS; 
        CRH.C = self.Cm * pF; 
        CRH.taum = self.taum *ms;
        CRH.gL = (self.Cm * pF) / (self.taum * ms); 
        CRH.EL = self.EL* mV;
        CRH.VT = self.VT * mV;
        CRH.Ea = self.Ea * mV
        CRH.DeltaA = (self.DeltaA) * mV 
        CRH.Vcut = (self.VT + 5 * self.DeltaT) * mV;
        CRH.refrac = self.refractory*ms;
        CRH.DeltaT = self.DeltaT * mV; 
        CRH.Va = self.Va * mV
        CRH.VR = self.VR *mV
        

        # init
        P.v = self.EL * mV
        

        # monitor
        M = SpikeMonitor( CRH )
        V = StateMonitor( CRH, ["v", "w"], record=True, dt=self.dt * ms)
        run(self._run_time * second)
        return M,V
       