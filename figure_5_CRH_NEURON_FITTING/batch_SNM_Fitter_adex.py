"""
This is the primary fitting script
"""
from loadABF import *
from loadNWB import *
from utils import *
import os
import glob
import pandas as pd
from scipy import stats
from multiprocessing import freeze_support
from joblib import Parallel, delayed
import single_neuron_spike_and_trace_fitter_v2_grid_with_refrac as snm_fit

#points to the folder containing the nwb files
FOLDER_PATH = "C:\\Users\\SMest\\Dropbox\\inoue_snm\\data\\HYP_CELL_NWB//Naive//"

#points to a file containing the cell ids and precomputed membrane parameters (cm, taum), if you want to use precomputed parameters
#current uneeded
membrane_params = "C:\\Users\\SMest\\Dropbox\\inoue_snm\\data\\old\\neuron_model_membrane_params.csv"
#points to a file containing the cell ids and precomputed QC for the cell. EG. whether cell passes or fails
qc = "C:\\Users\\SMest\\Dropbox\\inoue_snm\\data\\old\\qc.csv"
#points to the folder containing the output of the fitting script. Checks to see if the cell has already been fit and if not, fits it
pre_fit_check = glob.glob("C:\\Users\\SMest\\Dropbox\\inoue_snm\\code\\output\\adexfits\\realistic_CM_REFRAC_LARGE_DELTAT\\*_spike_fit_opt_CSV.csv", recursive=True)
pre_ids = [x.split("\\")[-1].split("_s")[0] for x in pre_fit_check]
pre_ids = [x.split("_c")[0] for x in pre_ids]
parallel = False
test = 'SM-A1-C11(111)-P08-16607002'
mem_p = pd.read_csv(membrane_params,  index_col=0)
qc_p = pd.read_csv(qc,  index_col=0)
mean_rms_filt = qc_p['Mean RMS'].to_numpy() <= 3
mean_drift_filt = qc_p['Mean Drift'].to_numpy() <= 5
filtered_ = np.logical_and(mean_rms_filt, mean_drift_filt)
id_qc = qc_p.index.values[filtered_]
id_ = mem_p.index.values


def fit_cell(fp, bypass_checks=True):
    """Primairy fitting function. Takes the path to the nwb file and returns a dataframe with the results of the fit.

    Args:
        fp (str): the path to the nwb file
        bypass_checks (bool, optional): Whether to bypass checks for the membrane parameters and the file qc. Defaults to True.

    Returns:
        fit_df: the fit results
    """
    cell_id = fp.split("\\")[-1].split(".")[0]
    realX, realY, realC, _ = loadNWB(fp)
    if realX.shape[0] == 15:
        if (bypass_checks==False) and (np.any(cell_id==id_) and np.any(cell_id==id_qc)):
            

                cell_params = mem_p.loc[cell_id].to_numpy()
                spikes = detect_spike_times(realX, realY, realC)
                cm = cell_params[0]
                taum = cell_params[2] * 1000
                if len(spikes[12]) > 3:
                    if cell_id in pre_ids:
                        print(f"Cell Id {cell_id} previously fit - Loading previous results")
                        try:
                            temp_df = pd.read_csv(f"C:\\Users\\SMest\\Dropbox\\inoue_snm\\code\\output\\adexfits\\realistic_CM_REFRAC_LARGE_DELTAT\\{cell_id}_spike_fit_opt_CSV.csv")
                        except:
                            temp_df = pd.read_csv(f"C:\\Users\\SMest\\Dropbox\\inoue_snm\\code\\output\\adexfits\\realistic_CM_REFRAC_LARGE_DELTAT\\{cell_id}_checkpoint_spike_fit_opt_CSV.csv")
                    else:
                        
                        temp_df = snm_fit.run_optimizer(fp, cm, taum, rounds_=150, batch_size_=500, optimizer='ng')
                    temp_df['id'] = [cell_id]
                    temp_df['cm_comp'] = [cm]
                    temp_df['taum_comp'] = [taum]
                    return temp_df
            
                print("fail at spike")
                return pd.DataFrame()
        elif bypass_checks==True:
                spikes = detect_spike_times(realX, realY, realC)
                cm = 10
                taum = 10
                if len(spikes[12]) > 3:
                    if cell_id in pre_ids:
                        print(f"Cell Id {cell_id} previously fit - Loading previous results")
                        try:
                            temp_df = pd.read_csv(f"C:\\Users\\SMest\\Dropbox\\inoue_snm\\code\\output\\adexfits\\realistic_CM_REFRAC_LARGE_DELTAT\\{cell_id}_spike_fit_opt_CSV.csv")
                        except:
                            temp_df = pd.read_csv(f"C:\\Users\\SMest\\Dropbox\\inoue_snm\\code\\output\\adexfits\\realistic_CM_REFRAC_LARGE_DELTAT\\{cell_id}_checkpoint_spike_fit_opt_CSV.csv")
                    else:
                        
                        temp_df = snm_fit.run_optimizer(fp, cm, taum, rounds_=150, batch_size_=500, optimizer='ng')
                    temp_df['id'] = [cell_id]
                    temp_df['cm_comp'] = [cm]
                    temp_df['taum_comp'] = [taum]
                    return temp_df
            
                print("fail at spike")

def main():
    _path = glob.glob(FOLDER_PATH +'*.nwb')
    #_path = glob.glob(_dir +'//..//data//HYP_CELL_NWB//RRS//*.nwb')
    full_df = pd.DataFrame()
    if parallel:
        dataframes =  Parallel(n_jobs=4, backend='multiprocessing')(delayed(fit_cell)(fp) for fp in _path)
    else:
        for fp in _path:
                temp = fit_cell(fp)
                full_df = full_df.append(temp)
    full_df.to_csv(f'output//full_spike_fit.csv')


if __name__ == "__main__": 
    freeze_support()
    main()