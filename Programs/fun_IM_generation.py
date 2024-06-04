#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antoniosquicciarini
KAI - ZEN
"""
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.integrate import quad, simps, trapz
import matplotlib.pyplot as plt

def generate_mean_function(pdf_list):

    n_pdfs = len(pdf_list)

    if n_pdfs == 0:
        raise ValueError("pdf_list must contain at least one PDF function")

    def mean_pdf_func(x):
        result = 0
        for pdf in pdf_list:
            result += pdf(x)
        return result/n_pdfs

    return mean_pdf_func

    
def add_epsilon(pdf, eps):
    def pdf_corrected(x):
        return pdf(x) + eps
    return pdf_corrected

def num_der(func, x, h=1e-12):
    # Compute the central difference for the derivative
    return (func(x + h) - func(x - h)) / (2 * h)

# def shannon_entropy(pdf, x):
#     print(simps(pdf(x),x))
#     pdf_norm = pdf(x)/simps(pdf(x), x)
#     print(simps(pdf_norm,x))
#     return -simps(np.nan_to_num(pdf_norm*np.log(pdf_norm), nan=0), x) 

def shannon_entropy(pdf, x):
    #print(simps(pdf(x),x))
    return -simps(pdf(x)*np.log(pdf(x)), x) 

def renyi_entropy(pdf, x, alpha):
    if alpha == 1:
        return shannon_entropy(pdf, x)
    else:
        return 1 / (1 - alpha) * np.log(simps(pdf(x)**alpha, x))

def tsallis_entropy(pdf, x, q):
    if q == 1:
        return shannon_entropy(pdf, x)
    else:
        return 1 / (q - 1) * (1 - simps(pdf(x)**q, x))
    
def fisher_information(pdf, x):
    return simps(num_der(pdf, x)**2/pdf(x))


def disequilibrium(pdf, x):
    return simps(pdf(x)**2)

def jensen_shannon_divergence(pdf_list, supports, args):

    pdf_mean_func = generate_mean_function(pdf_list) 
    x_min = np.min(supports, axis=0)[0]
    x_max = np.max(supports, axis=0)[1]
    x = np.linspace(x_min, x_max, args.x_discr) # to guarantee a support big enough to include ALL the
    JSD_1 = shannon_entropy(pdf_mean_func, x)
    JSD_2 = 0

    for ii, pdf in enumerate(pdf_list):
        x = np.linspace(supports[ii, 0], supports[ii, 1], args.x_discr) # to guarantee a support big enough to include ALL the
        JSD_2 += shannon_entropy(pdf, x)

    JSD_2 /= len(pdf_list)     
    
    return JSD_1-JSD_2
    
def jensen_fisher_divergence(pdf_list, x):
    pdf_mean_func = generate_mean_function(pdf_list) 
    JFD_1 = fisher_information(pdf_mean_func, x)
    JFD_2 = 0
    for pdf in pdf_list:
        JFD_2 += fisher_information(pdf, x)

    JFD_2 /= len(pdf_list)                      
    return -JFD_1+JFD_2


def TDE_computation(pdfs_list, supports_list, window_label_lists, args, indx):
    IMs_lists = []
    for jj, pdfs in enumerate(pdfs_list):
        IMs_list = []
        IMs = []
        with tqdm(total=len(pdfs), desc=f"Generating TDE: Delta {args.DeltaX_h_comb_list[jj][0][0]}") as pbar:
            DeltaX, _ = args.DeltaX_h_comb_list[jj][0]
            for ii, pdf in enumerate(pdfs):
                t_i = (ii * args.deltaX + DeltaX) / args.sampling_rate
                IMs.append(t_i)
                IMs.append(window_label_lists[jj][ii])

                anomaly_intensity = (t_i - args.t0_anomaly) / (args.t_total+1E-12 - args.t0_anomaly)
                if anomaly_intensity <= 0.0:
                    IMs.append(0.0)
                elif anomaly_intensity > 0.0:
                    IMs.append(anomaly_intensity)

                x = np.linspace(supports_list[jj][ii, 0], supports_list[jj][ii, 1], args.x_discr)

                for IM_indx in args.IM_list:
                    if "S" in IM_indx:
                        IMs.append(shannon_entropy(pdf, x))
                    elif "T" in IM_indx:
                        q = float(IM_indx[2:])
                        IMs.append(tsallis_entropy(pdf, x, q))
                    elif "R" in IM_indx:
                        q = float(IM_indx[2:])
                        IMs.append(renyi_entropy(pdf, x, q))
                    elif "F" in IM_indx:
                        IMs.append(fisher_information(pdf, x))

                IMs_list.append(IMs)
                IMs = []
                pbar.update(1)

        IMs_lists.append(IMs_list)

    args.IM_list.insert(0, 't') #time reference
    args.IM_list.insert(1, 'l') #anomaly label
    args.IM_list.insert(2, 'a') #anomaly intensity

    # Find the maximum length
    max_length = max(len(lst) for lst in pdfs_list)
    N_elements = len(IMs_lists[0][0])
    # Pad lists with zeros at the beginning if needed. Due to the different time references related with differend windows lengths
    
    n_min = min([len(IMs_list) for IMs_list in IMs_lists])

    # Cut the first part of shorter Delta windows to sincronize all of them
    for IM_list in IMs_lists: 
        n_red = len(IM_list)-n_min
        del IM_list[:n_red]
        # while len(IM_list) < max_length:
        #     IM_list.insert(0, [0] * N_elements)
    TDE_np = np.array(IMs_lists)
    
    return TDE_np


def stat_momentum_calculator(pdfs_list, supports_list, window_label_lists, args, indx):
    stat_mom_lists = []
    for jj, pdfs in enumerate(pdfs_list):
        stat_mom_list = []
        with tqdm(total=len(pdfs), desc=f"Generating stat_mom: Delta {args.DeltaX_h_comb_list[jj][0][0]}") as pbar:
            DeltaX, _ = args.DeltaX_h_comb_list[jj][0]
            for ii, pdf in enumerate(pdfs):

                x = np.linspace(supports_list[jj][ii, 0], supports_list[jj][ii, 1], args.x_discr)

                # Define the integrands for mean, variance, skewness, and kurtosis
                mean_integrand = lambda x: x * pdf(x)
                variance_integrand = lambda x, mean: (x - mean)**2 * pdf(x)
                skewness_integrand = lambda x, mean: (x - mean)**3 * pdf(x)
                kurtosis_integrand = lambda x, mean: (x - mean)**4 * pdf(x)

                # Calculate and store the statistical moments
                mean = simps(mean_integrand(x), x)
                stat_mom = [
                    mean,
                    simps(variance_integrand(x, mean), x),
                    simps(skewness_integrand(x, mean), x),
                    simps(kurtosis_integrand(x, mean), x)
                ]

                stat_mom_list.append(stat_mom)
                pbar.update(1)

        stat_mom_lists.append(stat_mom_list)

    n_min = min([len(stat_mom_list) for stat_mom_list in stat_mom_lists])

    # Cut the first part of shorter Delta windows to sincronize all of them
    for stat_mom_list in stat_mom_lists: 
        n_red = len(stat_mom_list)-n_min
        del stat_mom_list[:n_red]

    stat_mom_lists = np.array(stat_mom_lists)
    
    return stat_mom_lists






















