#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antoniosquicciarini
KAI - ZEN
"""
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats, optimize

from scipy.integrate import quad, simps, trapz
from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold
from sklearn.neighbors import KernelDensity

from fun_IM_generation import jensen_shannon_divergence
from fun_dataset_methods import extract_window

"""
Ref: https://medium.com/@BillChen2k/kernel-density-estimation-with-python-from-scratch-c200b187b6c4
"""

def scott_rule(data: np.ndarray):
    n = len(data)
    s = np.std(data, axis=0, ddof=1)
    return 3.49 * s * n ** (-0.333)

def silverman_rule(data: np.ndarray):
    def _select_sigma(x):
        normalizer = 1.349
        iqr = (stats.scoreatpercentile(x, 75) - stats.scoreatpercentile(x, 25)) / normalizer
        std_dev = np.std(x, axis=0, ddof=1)
        return np.minimum(std_dev, iqr) if iqr > 0 else std_dev
    sigma = _select_sigma(data)
    n = len(data)
    return 0.9 * sigma * n ** (-0.2)

def cross_validation_bandwidth(data: np.ndarray, k):
    """
    Ref: https://rdrr.io/cran/kedd/src/R/MLCV.R
    """
    n = len(data)
    x = np.linspace(np.min(data), np.max(data), n)
    def mlcv(h):
        fj = np.zeros(n)
        for j in range(n):
            for i in range(n):
                if i == j: continue
                fj[j] += k((x[j] - data[i]) / h)
            fj[j] /= (n - 1) * h
        return -np.mean(np.log(fj[fj > 0]))
    h = optimize.minimize(mlcv, 1)
    if np.abs(h.x[0]) > 10:
        return scott_rule(data)
    return h.x[0]



def cross_validation_bandwidth(data, bandwidths, cv_method='LOO'):
    kde = KernelDensity(kernel='gaussian')
    if cv_method == 'LOO':
        cv = LeaveOneOut()
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(kde, {'bandwidth': bandwidths}, cv=cv)
    grid.fit(data.reshape(-1, 1))
    return grid.best_estimator_.bandwidth

# Plot the KDE for different bandwidths
def plot_kde(data, bandwidth, title):
    x = np.linspace(min(data) - 1, max(data) + 1, 1000).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data.reshape(-1, 1))
    log_dens = kde.score_samples(x)
    
    plt.plot(x, np.exp(log_dens), label=f'Bandwidth: {bandwidth:.2f}')
    plt.title(title)
    plt.legend()

def pdf_specific_h_opt_methods(dataset, args):
    
    DeltaX_list = args.grid_exp_base**np.arange(args.DeltaX_grid[0], args.DeltaX_grid[1])

    #supports = []
    args.DeltaX_h_comb_list = []
        
    for DeltaX in DeltaX_list:
        DeltaX_h_comb_list_ii = []
        N_winddows_h_opt = (args.N_points - DeltaX)//args.deltaX 

        for ii in range(int(N_winddows_h_opt)+1):
            data = extract_window(dataset, args.deltaX, DeltaX, ii)

            if 'Silverman' in args.h_opt_strategy:
                h = silverman_rule(data)
            elif 'Scott' in args.h_opt_strategy:
                h = scott_rule(data)
            #supports.append((min(data)-4*h, max(data)+4*h))

            DeltaX_h_comb_list_ii.append((DeltaX, h))

        args.DeltaX_h_comb_list.append(DeltaX_h_comb_list_ii)

    return args

