import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) #trying to log brian2 errors
import os
from brian2 import *
from loadNWB import *
from utils import *
import pandas as pd
from joblib import dump, load
from b2_model.error import weightedErrorMetric
import nevergrad.common.typing as tp
import nevergrad as ng
from skopt import Optimizer, space, plots
TBPSAwithHam = ng.optimizers.Chaining([ng.optimizers.ScrHammersleySearch, ng.optimizers.NaiveTBPSA], ["num_workers"])
DEwithHam = ng.optimizers.Chaining([ng.optimizers.ScrHammersleySearch, ng.optimizers.TwoPointsDE], ["num_workers"])
class snmOptimizer():
    ''' A backend agnostic optimizer, which should allow easier switching between skopt and nevergrad. internal optimizaion code handles returning the 
    params in a way that b2 models want'''
    def __init__(self, params, param_labels, batch_size, rounds, backend='ng', nevergrad_kwargs={}, skopt_kwargs={}, nevergrad_opt=DEwithHam):
        if backend == 'ng':
            self.opt = _internal_ng_opt(params, param_labels, batch_size, rounds, nevergrad_opt, nevergrad_kwargs=nevergrad_kwargs)
            self.ask = self.opt.ask
            self.tell = self.opt.tell
            self.get_result = self.opt.get_result
        elif backend == 'skopt':
            self.opt = _internal_skopt(params, param_labels, batch_size, rounds)
            self.ask = self.opt.ask
            self.tell = self.opt.tell
            self.get_result = self.opt.get_result
    def ask():
        pass
    def tell():
        pass
    def get_result():
        pass

class _internal_ng_opt():
    def __init__(self, params, param_labels, batch_size, rounds, optimizer, nevergrad_kwargs={}):
        #Build Params
        self._params = params
        self._param_labels = param_labels
        self._build_params_space()
        #intialize the optimizer
        self.rounds = rounds
        self.batch_size = batch_size
        self.optimizer = optimizer
        budget = (rounds * batch_size)
        self.opt = self.optimizer(parametrization=self.params, num_workers=batch_size, budget=budget, **nevergrad_kwargs)
    def _build_params_space(self):
        ## Params should be a dict
        var_dict = {}
        for i, var in enumerate(self._params):
            temp_var = ng.p.Scalar(lower=var[0], upper=var[1], mutable_sigma=True) #Define them in the space that nevergrad wants
            var_dict[self._param_labels[i]] = temp_var
        self.params = ng.p.Dict(**var_dict)
    def ask(self, n_points=None):
        if n_points is None:
            n_points = self.batch_size
        self.points_list = []
        self.param_list =[]
        for p in np.arange(n_points):
                temp = self.opt.ask()
                self.param_list.append(temp.value)
                self.points_list.append(temp)
        param_list = pd.DataFrame(self.param_list)
            
        param_dict = param_list.to_dict('list')
        for key, value in param_dict.items():
            #Force into a numpy array
            param_dict[key] = np.asarray(value)
        return param_dict
    def tell(self, points, errors):
        #assume its coming back in with the same number of points
        #otherwise this will break
        assert errors.shape[0] == len(self.points_list)
        for i, row in enumerate(self.points_list):
            self.opt.tell(row, errors[i])
    def get_result(self):
        return self.opt.provide_recommendation().value
        
class _internal_skopt():
    def __init__(self, params, param_labels, batch_size, rounds, optimizer='RF', skopt_kwargs={}):
        #Build Params
        self._params = params
        self._param_labels = param_labels
        self._build_params_space()
        self.rounds = rounds
        self.batch_size = batch_size
        #intialize the optimizer
        self.opt = Optimizer(dimensions=self.params, base_estimator=optimizer, n_initial_points=self.batch_size, acq_optimizer='sampling', n_jobs=-1)

    def _build_params_space(self):
        self.params = space.Space(self._params)
    
    def ask(self, n_points=None):
        if n_points is None:
            n_points = self.batch_size
        self.param_list = np.asarray(self.opt.ask(n_points=n_points)) ##asks the optimizer for new params to fit
        self.param_dict = {}
        for col, label in zip(self.param_list.T, self._param_labels):
            self.param_dict[label] = col
        return self.param_dict
    def tell(self, points, errors):
        #assume its coming back in with the same number of points
        #otherwise this will break
        assert errors.shape[0] == len(self.param_list)
        self.opt.tell(self.param_list.tolist(), errors.tolist())
    def get_result(self):
        results = self.opt.get_result() #returns a result containing the param - error matches
        out = results.x ##asks the optimizer for new params to fit
        param_dict = {'Cm': out[0], 'taum':out[1], 'EL': out[2], 'a': out[3], 
                    'tauw': out[4], 'VT': out[5], 'VR': out[6],
                    'b': out[7]}
        return param_dict




   
