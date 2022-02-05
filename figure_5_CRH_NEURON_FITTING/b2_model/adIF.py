#
#  CRH single-neuron model
#  Inoue collaboration
#  19 feb 2020
#
# PERFORM ADAPTIVE LEAKY INTEGRATE-AND-FIRE SIMULATION USING BRIAN SOFTWARE
#
# INPUTS (ALL SI UNITS)
# C - membrane capacitance
# taum - membrane time constant
# a -
# b -
# folder - full path to output folder
# ii - iteration number
#
from utils import *
from brian2 import *
from brian2_model import brian2_model

def adIF(C, taum, a, b, folder, ii):
    from scipy.io import savemat
    from brian2 import nS, mV, ms, nA, mvolt, msecond, Equations, NeuronGroup, PoissonInput, randn, \
        SpikeMonitor, StateMonitor, run, array, Hz
    from os import mkdir
    from os.path import isdir
    import time

    start = time.time()
    print(f"{ii:d}, ")

    # simulation parameters
    N = int(1e4)

    # neuron parameters
    gL = C/taum
    EL = -70.6 * mV
    VT = -50.4 * mV  # threshold

    # synapse parameters
    Ee = 0 * mvolt
    Ei = -80 * mvolt
    taue = 5 * msecond
    taui = 5 * msecond
    we = 6. * nS
    wi = 67. * nS  # excitatory/inhibitory synaptic weight

    # Membrane Equation
    eqs = Equations('''
    dv/dt = ( gL*(EL-v) + ge*(Ee-v) + gi*(Ei-v) + I - w ) * (1./C) : volt
    dw/dt = ( a*(v - EL) - w ) / tauw : amp
    dge/dt = -ge*(1./taue) : siemens
    dgi/dt = -gi*(1./taui) : siemens
    tauw : second
    a : siemens
    b : amp
    I : amp''')

    # build network
    P = NeuronGroup(N, model=eqs, threshold='v>VT', reset='v=EL', refractory=5 * ms)
    CRH = P
    # CRH.tauw = LTS['tauw']
    # CRH.b = LTS['b']
    # CRH.a = linspace(0, 0, N) * nS
    CRH.tauw = tauw
    CRH.a = a
    CRH.b = b

    # poisson input
    input = PoissonInput(CRH, 'ge', 1, 10 * Hz, weight=we)

    # init
    P.v = randn(len(P)) * 2 * mV - 70 * mV
    P.ge = (randn(len(P)) * 2 + 5) * we
    P.gi = (randn(len(P)) * 2 + 5) * wi

    # monitor
    M = SpikeMonitor(CRH)
    V = StateMonitor(CRH, 'v', record=True)

    # run simulation
    CRH.I = -0.5 * nA
    run(100 * ms)
    CRH.I = 0.5 * nA
    run(100 * ms)
    CRH.I = 0 * nA
    run(800 * ms)

    st = array(M.t / ms)
    si = array(M.i)
    t = array(V.t / ms)
    vm = array(V.v / mV)

    if ~isdir(folder):
        mkdir(folder)
    savemat("folder/Cm=%s_tauF=%s_tau=%s_Rm=%s.mat" % (C, tauF, tau, Rm), mdict={'st': st, 'si': si, 't': t, 'vm': vm},
            oned_as='column')

    end = time.time() - start
    print(" | Iteration completion duration: %.2f\n" % end)

class adIFModel(brian2_model):
    ''' 
    Represents an adIF model object that can be called while storing important params that don't need to change.
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
        eqs = Equations('''
        dv/dt = ( gL*(EL-v) + I - w ) * (1./C) : volt (unless refractory)
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
        refrac : second
        I = in_current(t) : amp
        ''')


 
        # build network
        P = NeuronGroup( self.N, model=eqs, threshold='v>VT', reset='v=VR; w+=b', refractory=self.refractory*ms, method='euler')
        CRH = P; 
        CRH.tauw = self.tauw *ms; 
        CRH.b = self.b * nA; 
        CRH.a = self.a * nS; 
        CRH.C = self.Cm * pF; 
        CRH.taum = self.taum *ms;
        CRH.gL = (self.Cm * pF) / (self.taum * ms); 
        CRH.EL = self.EL* mV;
        CRH.VT = self.VT * mV;
        
        #CRH.Vcut = (self.VT + 5 * self.DeltaT) * mV;
        CRH.refrac = self.refractory*ms;
         
        
        CRH.VR = self.VR *mV
        

        # init
        P.v = self.EL * mV
        # monitor
        M = SpikeMonitor( CRH )
        V = StateMonitor( CRH, ["v", "w"], record=True )
        run(self._run_time * second)
        return M,V
        
    