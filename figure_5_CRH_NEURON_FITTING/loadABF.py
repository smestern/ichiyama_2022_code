import numpy as np
import pyabf

def loadABF(file_path, return_obj=False):
    '''
    Employs pyABF to generate numpy arrays of the ABF data. Optionally returns abf object.
    Same I/O as loadNWB
    '''
    abf = pyabf.ABF(file_path)
    dataX = []
    dataY = []
    dataC = []
    for sweep in abf.sweepList:
        abf.setSweep(sweep)
        tempX = abf.sweepX
        tempY = abf.sweepY
        tempC = abf.sweepC
        dataX.append(tempX)
        dataY.append(tempY)
        dataC.append(tempC)
    npdataX = np.vstack(dataX)
    npdataY = np.vstack(dataY)
    npdataC = np.vstack(dataC)

    if return_obj == True:

        return npdataX, npdataY, npdataC, abf
    else:

        return npdataX, npdataY, npdataC

    ##Final return incase if statement fails somehow
    return npdataX, npdataY, npdataC

