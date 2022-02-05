import numpy as np
try:
    import h5py
    ##Does not import when using python-matlab interface on windows machines
except:
    print("h5py import fail")
import pandas as pd


def loadNWB(file_path, return_obj=False):
    ''' Loads the nwb object and returns three arrays dataX, dataY, dataC and optionally the object.
    same input / output as loadABF for easy pipeline inclusion
    
    dataX - time (should be seconds)
    dataY - voltage (should be mV)
    dataC - current (should be pA)
    dt - sampling interval (should be seconds)
    '''
    nwb = nwbFile(file_path)
    
    fs_dict = nwb.rate # sampling rate info
    fs = fs_dict["rate"] # assumes units of Hz
    dt = np.reciprocal(fs) # seconds
    
    if isinstance(nwb.dataX, np.ndarray)==False:
        dataX = np.asarray(nwb.dataX, dtype=np.dtype('O')) ##Assumes if they are still lists its due to uneven size
        dataY = np.asarray(nwb.dataY, dtype=np.dtype('O')) #Casts them as numpy object types to deal with this
        dataC = np.asarray(nwb.dataC, dtype=np.dtype('O'))
    else:
        dataX = nwb.dataX #If they are numpy arrays just pass them
        dataY = nwb.dataY
        dataC = nwb.dataC

    if return_obj == True:
        return dataX, dataY, dataC, dt, nwb
    else:
        return dataX, dataY, dataC, dt

    ##Final return incase if statement fails somehow
    return dataX, dataY, dataC, dt



# A simple class to load the nwb data quick and easy
##Call like nwb = nwbfile('test.nwb')
##Sweep data is then located at nwb.dataX, nwb.dataY, nwb.dataC (for stim)
class nwbFile(object):

    def __init__(self, file_path):
        with h5py.File(file_path,  "r") as f:
            ##Load some general properities
            sweeps = list(f['acquisition'].keys()) ##Sweeps are stored as keys
            self.sweepCount = len(sweeps)
            self.rate = dict(f['acquisition'][sweeps[0]]['starting_time'].attrs.items())
            self.sweepYVars = dict(f['acquisition'][sweeps[0]]['data'].attrs.items())
            self.sweepCVars = dict(f['stimulus']['presentation'][sweeps[0]]['data'].attrs.items())
            ##Load the response and stim
            data_space_s = 1/self.rate['rate']
            dataY = []
            dataX = []
            dataC = []
            for sweep in sweeps:
                temp_dataY = np.asarray(f['acquisition'][sweep]['data'][()])
                temp_dataX = np.cumsum(np.hstack((0, np.full(temp_dataY.shape[0]-1,data_space_s))))
                temp_dataC = np.asarray(f['stimulus']['presentation'][sweep]['data'][()])
                dataY.append(temp_dataY)
                dataX.append(temp_dataX)
                dataC.append(temp_dataC)
            try:
                ##Try to vstack assuming all sweeps are same length
                self.dataX = np.vstack(dataX)
                self.dataC = np.vstack(dataC)
                self.dataY = np.vstack(dataY)
            except:
                #Just leave as lists
                self.dataX = dataX
                self.dataC = dataC
                self.dataY = dataY
        return

