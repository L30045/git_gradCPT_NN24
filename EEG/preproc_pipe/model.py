
#%% load library
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne
mne.viz.set_browser_backend("matplotlib")
import os
import pickle
import glob
import time
import sys
import statsmodels.api as sm
import pandas as pd
from patsy import dmatrices



#%% helper function
def make_design_matrix(X, winlen=None):
    # flat time series
    X = X.flatten()
    # assign IRF window length
    if not winlen:
        winlen = len(X)
    # create design_matrix
    design_matrix = []
    for t_i in range(winlen):
        shift_S = np.concatenate([np.zeros(t_i), X[:len(X)-t_i]])
        design_matrix.append(shift_S)
    design_matrix = np.stack(design_matrix,axis=1)
    return design_matrix