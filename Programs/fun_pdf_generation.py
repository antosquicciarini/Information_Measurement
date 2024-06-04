#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antoniosquicciarini

14 March 2022

KAI ZEN
"""

#%% GENERAL SETTINGS
# Packages to import

import numpy as np
from fun_KDE_transformations import JSD_time_serie_h_opt, add_epsilon
from fun_KDE_transformations import pdf_list_generation as KDE_pdf_list_generation
from fun_KDE_other_h_opt_methods import pdf_specific_h_opt_methods
from fun_dataset_methods import extract_window, count_N_windows
from scipy.stats import gaussian_kde
from fun_KDE_transformations import kde

"""
The program generates data ONLY about seizure affected records 
"""

def window_extraction(args, dataset, ii):
    return dataset[args.deltaX*ii : args.DeltaX+args.deltaX*ii, :]

def pdf_generation(dataset, args, indx): #path_database, chdir, obj_dir, add_dir, DeltaX, deltaX, prob_distr_type, beta, Q_list, chb_indx, h_opt_strategy, KDE_eps, h_cost, min_Pre_Ict, min_Post_Ict, norm_flag, freq_bands):

    args.N_points = dataset.shape[0]
    if args.prob_distr_type == 'KDE' and 'JSD' in args.h_opt_strategy:
        args = JSD_time_serie_h_opt(dataset, args) 
        
    elif args.prob_distr_type == 'KDE' and 'cost' in  args.h_opt_strategy:

        args.DeltaX_h_comb_list = []
        for Delta_h_pair in args.h_Delta_list:
            N_windows = count_N_windows(args.N_points, Delta_h_pair[0], args.deltaX)
            args.DeltaX_h_comb_list.append([Delta_h_pair for jj in range(N_windows)])

    else:
        args = pdf_specific_h_opt_methods(dataset, args)


    pdfs_list, supports_list, window_labels_list = [], [], []

    if args.prob_distr_type == 'KDE':
        for DeltaX_h_comb_list_ii in args.DeltaX_h_comb_list:
            pdfs, supports, window_labels = [], [], []

            for w_ii, (DeltaX, h) in enumerate(DeltaX_h_comb_list_ii):

                data = extract_window(dataset, args.deltaX, DeltaX, w_ii)
                pdf = kde(data, h)
                eps_pdf = 1E-12 #Little value of probability over all the domain

                window_label = 1 + (args.deltaX*w_ii - args.t0_anomaly*args.sampling_rate) / (DeltaX)
                if window_label<0:
                    window_label = 0.0
                elif window_label>1:
                    window_label = 1.0
                
                pdfs.append(add_epsilon(pdf, eps_pdf)) 
                supports.append((min(data)-4*h, max(data)+4*h))
                window_labels.append(window_label)

            pdfs_list.append(pdfs)
            supports_list.append(np.array(supports))
            window_labels_list.append(window_labels)

    return pdfs_list, supports_list, window_labels_list
