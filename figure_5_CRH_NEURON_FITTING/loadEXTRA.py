import numpy as np
import os
from utils import *
file_dir = os.path.dirname(__file__)
isi_path = file_dir +'//..//data//aoi_data//extra//ISI_series.csv'
time_path = file_dir +'//..//data//aoi_data//extra//TIME_series.csv'

def loadISI(unit=None):
    '''Takes:
        unit (optional): the unit number to load, otherwise random
        ---
        Returns:
        array of isis'''
    if unit is None:
        unit = np.random.randint(1,36) #if unit is not specified just grab a random one
    unit = unit-1 ##accounting for 0 indexing in python
    ISI_ar = np.genfromtxt(isi_path, delimiter=",")[:,unit] #load that unit
    ISI_ar = ISI_ar[~np.isnan(ISI_ar)] #drop the nans
    return ISI_ar

def loadTIME(unit=None):
    if unit is None:
        unit = np.random.randint(1,36) #if unit is not specified just grab a random one
    unit = unit-1 ##accounting for 0 indexing in python
    TIME_ar = np.genfromtxt(time_path, delimiter=",")[:,unit] #load that unit
    TIME_ar = TIME_ar[~np.isnan(TIME_ar)]
    if unit == 8:
        TIME_ar = np.cumsum(loadISI(9)) / 1000
        
     #drop the nans
    return TIME_ar

