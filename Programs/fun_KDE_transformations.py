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
from scipy.stats import gaussian_kde, norm
from sklearn.neighbors import KernelDensity

from scipy.integrate import quad, simps, trapz
import matplotlib
from fun_IM_generation import jensen_shannon_divergence
from fun_dataset_methods import extract_window, count_N_windows

def generate_mean_function(pdf_list):
    """
    pdf_list: List of PDF functions, e.g., [kde1, kde2, ...]
    weights: List of weights for each PDF, e.g., [weight1, weight2, ...]
    x_range: Tuple representing the range of the variable, e.g., (x_min, x_max)
    """
    # Define the mean PDF calculation function
    def mean_pdf_func(x):
        result = 0
        for pdf in pdf_list:
            result += pdf(x)
        return result/len(pdf_list)

    return mean_pdf_func

def kde(X, h):
    kde = KernelDensity(bandwidth=h, kernel='gaussian')
    kde.fit(X.reshape(-1, 1))

    def pdf(x_values):
        return np.exp(kde.score_samples(x_values.reshape(-1, 1)))
    
    return pdf

def add_epsilon(pdf, eps):
    def pdf_corrected(x):
        return pdf(x) + eps
    return pdf_corrected

def num_der(func, x, h=1e-12):
    # Compute the central difference for the derivative
    return (func(x + h) - func(x - h)) / (2 * h)


def shannon_entropy(pdf, x):
    return -simps(pdf(x)*np.log(pdf(x)), x)

def fisher_information(pdf, x):
    return simps(num_der(pdf, x)**2/pdf(x))


def disequilibrium(pdf, x):
    return simps(pdf(x)**2)

    
def jensen_fisher_divergence(pdf_list, x):
    pdf_mean_func = generate_mean_function(pdf_list) 
    JFD_1 = fisher_information(pdf_mean_func, x)
    JFD_2 = 0
    for pdf in pdf_list:
        JFD_2 += fisher_information(pdf, x)

    JFD_2 /= len(pdf_list)                      
    return -JFD_1+JFD_2

def pdf_list_generation(dataset, args, DeltaX=None, h=None, t_stop=None):

    if DeltaX is None:
        DeltaX = args.DeltaX
    if h is None:
        h = args.h_opt

    if t_stop is not None:
        N_winddows_h_opt = (t_stop - DeltaX)//args.deltaX 
    else:
        N_winddows_h_opt = (args.N_points - DeltaX)//args.deltaX 

    pdf_list = []
    supports = []
    window_labels_list = []
    for ii in range(int(N_winddows_h_opt)+1):

        data = extract_window(dataset, args.deltaX, DeltaX, ii)
        supports.append((min(data)-4*h, max(data)+4*h))

        pdf = kde(data, h)
        #pdf = gaussian_kde(data, bw_method=h) #TOO fast, I was not able to create my solution
        
        eps_pdf = 1E-12 #Little value of probability over all the domain
        pdf_list.append(add_epsilon(pdf, eps_pdf)) 

        window_label = 1 + (args.deltaX*ii - args.t0_anomaly*args.sampling_rate) / (DeltaX)
        if window_label<0:
            window_label = 0.0
        elif window_label>1:
            window_label = 1.0
        window_labels_list.append(window_label)

    supports = np.array(supports)
    return pdf_list, supports, window_labels_list
        

def compute_quality_score(pdf_list, supports, args):

    if "PL" in args.h_opt_strategy:
        result = np.ones((len(pdf_list[0])))
        for pdf in pdf_list:
            result *= pdf
        result = np.mean(result)
        return result


    elif "JSD" in args.h_opt_strategy:
        return jensen_shannon_divergence(pdf_list, supports, args)

    elif "JFD" in args.h_opt_strategy:
        return jensen_fisher_divergence(pdf_list, supports)
    

def find_plateau_value_DeltaX(quality_score_df):
    DeltaX_h_comb_list = []

    for h_indx in quality_score_df.columns:
        col = quality_score_df[h_indx]
        prev_value = None
        plateau_value = None
        DeltaX_indx = None

        for DeltaX_indx, ele in col.items():
            if prev_value is not None:
                decrease_rate = (prev_value - ele)/prev_value
                if decrease_rate <= -0.01:
                    plateau_value = ele
                    min_idx = (col - prev_value).abs().idxmin()
                    DeltaX_h_comb_list.append((min_idx, h_indx))
                    break
            prev_value = ele

        if plateau_value is None:
            if DeltaX_indx is not None:
                DeltaX_h_comb_list.append((DeltaX_indx, h_indx))

    return DeltaX_h_comb_list


def find_plateau_value_diff_h(quality_score_df, threshold):
    DeltaX_h_comb_list = []

    quality_score_pct = quality_score_df.pct_change(axis=1)
    for DeltaX_indx in quality_score_pct.index:
        row = quality_score_pct.loc[DeltaX_indx]
        prev_h_indx = None
        for h_indx, ele in row.items():
            if np.abs(ele) > threshold:
                if prev_h_indx is None:
                    prev_h_indx = h_indx
                DeltaX_h_comb_list.append((DeltaX_indx, prev_h_indx))
                break
            else:
                prev_h_indx = h_indx
    return DeltaX_h_comb_list

    # variance_df = args.quality_score_df.diff(axis=1)
    # max_abs_values = variance_df.abs().max(axis=1)
    # normalized_variance_df = variance_df.divide(max_abs_values, axis=0)
    # df_plot(normalized_variance_df, f'{args.folder_path}/{args.exp_name}_JSD_quality_score_h_diff.png', highlight_cells=args.JSD_h_diff_plato)



def plot_DeltaX_h_quality_score_2D_surface(DeltaX_list, h_list, quality_score):

    # Create a meshgrid from the axes
    H, DeltaX = np.meshgrid(h_list, DeltaX_list)

    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D surface
    ax.plot_surface(H, DeltaX, quality_score, cmap='viridis')

    # Set labels for the axes
    ax.set_xlabel(r'$h$')
    ax.set_ylabel(r'$\Delta$')
    ax.set_zlabel(r'$JSD$')

    # Show the plot
    plt.show()

def DeltaX_h_comb_max_complexity(DeltaX_h_comb_list, dataset, channel, args, t_start_anomaly):
    Complexity_mean_score = []
    Fisher_mean_score = []
    Shannon_mean_score = []
    
    for DeltaX_h_comb in DeltaX_h_comb_list:
        pdf_list, supports, window_labels_list  = pdf_list_generation(dataset, channel, args, DeltaX=DeltaX_h_comb[0], h=DeltaX_h_comb[1], t_stop=t_start_anomaly)

        Complexity = []
        Fisher = []
        Shannon = []
        for ii, pdf in enumerate(pdf_list):
            x = np.linspace(supports[ii, 0], supports[ii, 1], args.x_discr) # to guarantee a support big enough to include ALL the
            Complexity.append(disequilibrium(pdf, x) * np.exp(2*shannon_entropy(pdf, x)) /(2*np.pi*np.e))

            Fisher.append(fisher_information(pdf,x))
            Shannon.append(shannon_entropy(pdf,x))

        Complexity_mean_score.append(np.mean(np.array(Complexity)))
        Fisher_mean_score.append(np.mean(np.array(Fisher)))
        Shannon_mean_score.append(np.mean(np.array(Shannon)))
    
    best_comb_indx = np.argmax(np.array(Complexity_mean_score))
            
    return DeltaX_h_comb_list[best_comb_indx]


def JSD_time_serie_h_opt(dataset, args):

    #optimize h over "healthy" time signal (ASSUMPTION: we know that a part of the signal is good)
    DeltaX_list = args.grid_exp_base**np.arange(args.DeltaX_grid[0], args.DeltaX_grid[1])

    t_stop = args.t0_anomaly * args.sampling_rate

    # Set initial interval and tolerance
    h_min_initial = 0.0001  # Set initial lower bound
    h_max_initial = 1.0 # Set initial upper bound
    epsilon = 1e-2  # Set tolerance

    args.DeltaX_h_comb_list_JSD  = []
    t_start_anomaly = args.t0_anomaly * args.sampling_rate

    def find_h(target_value, h_min, h_max, epsilon, dataset, args, DeltaX_ii, t_start_anomaly):
        while h_max - h_min > epsilon:
            h_mid = (h_min + h_max) / 2

            t_start_anomaly = args.t0_anomaly * args.sampling_rate
            pdf_list, supports, window_labels_list = pdf_list_generation(dataset, args, DeltaX=DeltaX_ii, h=h_mid, t_stop=t_start_anomaly)
            JDS_mid =  compute_quality_score(pdf_list, supports, args)

            if JDS_mid > target_value:
                h_min = h_mid
            else:
                h_max = h_mid

        return (h_min + h_max) / 2
    
    
    for ii, DeltaX_ii in enumerate(tqdm(DeltaX_list, desc='applying JSD-h algorithm...')):
        JD_max_value = np.log((t_stop - DeltaX_ii) // args.deltaX)
        target_value = JD_max_value * args.h_plato_th
        h_opt = find_h(target_value, h_min_initial, h_max_initial, epsilon, dataset, args, DeltaX_ii, t_start_anomaly)
        args.DeltaX_h_comb_list_JSD.append([DeltaX_ii, h_opt])

    args.DeltaX_h_comb_list = []
    for Delta_h_pair in args.DeltaX_h_comb_list_JSD :
        N_windows = count_N_windows(args.N_points, Delta_h_pair[0], args.deltaX) #check
        args.DeltaX_h_comb_list.append([Delta_h_pair for jj in range(N_windows)])

    return args


def plot_comparative_pdf_grid(dataset, DeltaX_list, h_list, args):

    border_color = 'green'
    border_thickness = 4

    fig, axes = plt.subplots(len(DeltaX_list), len(h_list), figsize=(14,  12)) #figsize=(45, 20)) 
    
    # Adjust spacing to reduce border size
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    for i, DeltaX in enumerate(DeltaX_list):
        for j, h in enumerate(h_list):
            
            pdf_list, supports, window_labels_list = pdf_list_generation(dataset, args, DeltaX=DeltaX, h=h)
            
            # Define the custom colormap going from blue (0) to red (1)
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list('BlueToRed', ['blue', 'red'])

            # Iterate through the pdf_list and supports
            for ii, pdf in enumerate(pdf_list):
                x = np.linspace(supports[ii, 0], supports[ii, 1], args.x_discr)

                # Get the corresponding color from the custom colormap based on the window_labels_list values
                line_color = cmap(window_labels_list[ii])

                # Plot the line with the specified color on the current axes
                axes[i, j].plot(x, pdf(x), color=line_color, alpha=0.01)

            # Create a color bar that goes from blue (0) to red (1) and add it to the current axes
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array([])
            

            # Add grid to the current axes
            axes[i, j].grid()
            if i == 0:
                axes[i, j].text(0.5, 1.1, rf'$h$-{h}', transform=axes[i, j].transAxes, fontsize=10, ha='center')

            if j == len(h_list)-1:
                plt.colorbar(sm, ax=axes[i, j])#, label='Anomaly Intensity')

            if j == 0:
                axes[i, j].text(-0.4, 0.5, rf'$\Delta$-{DeltaX}', transform=axes[i, j].transAxes, fontsize=10, va='center', rotation='vertical')

            if (DeltaX, h) in args.DeltaX_h_comb_list_JSD:
                for spine in axes[i, j].spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(border_thickness)

    # Save the entire figure
    plt.show()





