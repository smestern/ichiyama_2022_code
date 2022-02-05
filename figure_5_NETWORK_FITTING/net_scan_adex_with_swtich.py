

from multiprocessing import freeze_support
from utils import *
import time
from spike_train_utils import *
from loadEXTRA import loadISI
from skopt import dump, load
from net_scan_helper_adex_switching import network_scan
real_isi = loadISI(4)
from IPython.display import clear_output
#import nevergrad as ng
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import dill
import warnings
warnings.filterwarnings("ignore")
BrianLogger.suppress_hierarchy('brian2.codegen')
BrianLogger.suppress_hierarchy('brian2.groups.group.Group.resolve.resolution_conflict')

def main():

    # In[2]:


    from skopt import Optimizer, space
    param_space = space.Space([(0.2, 60), ##Ee
                            (0.2, 25.1), ##Ii
                                (2, 50),##de
                            (2, 1500),#di
                            (0.5, 200), #drch
                            (50,1500)])   #input hz
                            
    opt = Optimizer(dimensions=param_space, acq_optimizer="auto", n_initial_points=20, model_queue_size=12) ##Bayes optimizer
    #opt.tell(points.tolist(), dist.tolist())


    # In[5]:

    opt = load("opt.pkl")
    full_dist = []
    param_list_temp = []
    param_list = []
    for i in np.arange(200):
        print(f"Round {i} start")
        start_time = time.time()
        clear_output(wait=True)
        temp = opt.ask(n_points=4)
        for points in temp:
            print(points)
            network_scan(*points)
        y = Parallel(n_jobs=4, backend='multiprocessing')(delayed(network_scan)(*points) for points in temp)
        print(f"Round {i} Complete with min {np.amin(y)}")
        param_list.append(temp)
        opt.tell(temp, y)
        dump(opt, "opt.pkl")
        opt_results = opt.get_result()
        
        dump(opt_results, "opt_res.pkl")
        print(f"iter {i} finished {(time.time()-start_time)/60}")


    # In[ ]:


    opt_results = opt.get_result()
    


    # In[ ]:


    
    dump(opt_results, "opt_res.pkl")


    # In[ ]:


    from skopt.plots import plot_objective
    plot_objective(opt_results)
    print('d')

    # In[ ]:

def nevergrad():
    import nevergrad as ng
    

    var_dict = ng.p.Dict(
                        ##Network basic params
                        ee=ng.p.Scalar(lower=0.002, upper=15), 
                        ii=ng.p.Scalar(lower=0.002, upper=15),
                        wcrh=ng.p.Log(lower=0.0002, upper=40),
                        de=ng.p.Scalar(lower=0.1, upper=40),
                        di=ng.p.Scalar(lower=0.1, upper=40),
                        dcrh=ng.p.Scalar(lower=1, upper=600),
                        _vt=ng.p.Scalar(lower=-60, upper=-40),
                        _vr=ng.p.Scalar(lower=-80, upper=-40),
                       input_hz=ng.p.Log(lower=5,upper=1500),
                        p_ie=ng.p.Log(lower=0.001, upper=0.5),
                        p_ei=ng.p.Log(lower=0.001, upper=0.5),
                        ##Switch Params
                        i_pr=ng.p.Scalar(lower=0.1, upper=2.5),
                        e_pr=ng.p.Scalar(lower=0.1, upper=2.5),
                        b_change=ng.p.Log(lower=0.1, upper=100),
                        adaptation_time_constant=ng.p.Log(lower=0.1, upper=100),
                    )
    batch_size=4
    budget = 900
    #
    #TBPSAwithLHS = ng.optimizers.Chaining([ng.optimizers.ScrHammersleySearch, ng.optimizers.TwoPointsDE], [batch_size*4])
    opt = ng.optimizers.Portfolio(parametrization=var_dict, num_workers=batch_size, budget=budget*batch_size)
    #opt.tell(points.tolist(), dist.tolist())
    

    # In[5]:

    with open('optng2.pkl', 'rb') as file:
       opt = dill.load(file)
    full_dist = []
    param_list_temp = []
    param_list = []
    for i in np.arange(budget):
        print(f"Round {i} start")
        start_time = time.time()
        clear_output(wait=True)
        points = [] 
        points_ar = []
        print(f"iter {i} started {(time.time()-start_time)/60} - Now asking points")
        for x in np.arange(batch_size):
            temp = opt.ask()
            points.append(temp)
            points_ar.append(temp.value)

            #network_scan(**temp.value)
        try:
            print(f"iter {i} started {(time.time()-start_time)/60} - Now running")
            y = Parallel(n_jobs=batch_size, backend='multiprocessing')(delayed(network_scan)(**x) for x in points_ar)
            print(f"Round {i} Complete with min {np.amin(y)}")
            print(points[np.argmin(y)].value)
            print(f"iter {i} finished {(time.time()-start_time)/60} - Now Telling points")
            for i,x in enumerate(points):
                opt.tell(x, y[i])
            print(f"iter {i} finished {(time.time()-start_time)/60} - Now saving optimizer")
            with open('optng2.pkl', 'wb') as file:
                dill.dump(opt, file)
            print(f"iter {i} finished {(time.time()-start_time)/60} - Now clearing Cache")
            clear_cache('cython')
        except:
            with open('optng2.pkl', 'rb') as file:
                opt = dill.load(file)
        
        print(f"iter {i} finished {(time.time()-start_time)/60}")


    print('d')

    



   

if __name__ == '__main__':
    freeze_support()
    nevergrad()
    #main()


