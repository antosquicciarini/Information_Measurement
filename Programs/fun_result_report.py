#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antoniosquicciarini
KAI ZEN
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import os
from scipy.stats import zscore
import seaborn as sns
from datetime import datetime
import pymannkendall as mk
import pandas as pd
from fun_KDE_transformations import pdf_list_generation


def pdf_anomaly_plot(pdfs_list, supports_list, window_labels_list, args):
    # Define the custom colormap going from blue (0) to red (1)
    cmap = LinearSegmentedColormap.from_list('BlueToRed', ['blue', 'red'])
    fig, axes = plt.subplots(1, len(pdfs_list), figsize=(16,  2.5)) #figsize=(45, 20)) 
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Iterate through the pdfs_list and supports
    for Delta_ii, pdfs in enumerate(pdfs_list):
        for t_ii, pdf in enumerate(pdfs):
            x = np.linspace(supports_list[Delta_ii][t_ii, 0], supports_list[Delta_ii][t_ii, 1], args.x_discr)

            # Get the corresponding color from the custom colormap based on the window_labels_list values
            line_color = cmap(window_labels_list[Delta_ii][t_ii])

            # Plot the line with the specified color
            axes[Delta_ii].plot(x, pdf(x), color=line_color, alpha=1/np.log2(len(pdfs_list[0]))*0.15)
        axes[Delta_ii].grid()
        if "JSD" in args.h_opt_strategy:
            axes[Delta_ii].set_title(rf"$\Delta$:{args.DeltaX_h_comb_list[Delta_ii][0][0]}-$h$:{args.DeltaX_h_comb_list[Delta_ii][0][1]:.3f}")
        else: 
            h_mean = np.mean(np.array([DeltaX_h_comb[1] for DeltaX_h_comb in args.DeltaX_h_comb_list[Delta_ii]]))
            axes[Delta_ii].set_title(rf"$\Delta$:{args.DeltaX_h_comb_list[Delta_ii][0][0]}-$h_m$:{h_mean:.2f}")

    # Create a color bar that goes from blue (0) to red (1)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])

    # Add a color bar to your plot
    plt.colorbar(sm)
    plt.tight_layout()

    # Show the plot
    #plt.show()
    plt.savefig(f'{args.folder_path}/Delta_h_best_commb_pdf_evolution.png', dpi=300)
    plt.close()


def norm(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))


def IM_time_plots(TDE_np, dataset, args):

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    t_i = np.arange(len(dataset)) / args.sampling_rate
    axes.plot(t_i, dataset, color='blue', linewidth=0.05)
    axes2 = axes.twinx()  # Create a twin Axes sharing the xaxis

    axes2.plot(TDE_np[0, :, args.IM_list.index('t')], 
                    TDE_np[0, :, args.IM_list.index('a')], 
                    label="Anomaly Intensity", 
                    color='red')
    axes.legend(loc='upper right')

    axes.axvline(x=args.t0_anomaly, color='red', linestyle='--', label="Anomaly Time")

    axes.grid(True)
    axes.set_xlabel("time [s]")

    axes.set_xlim(min(t_i), max(t_i))
    axes.set_ylabel("amplitude")
    axes2.set_ylabel("anomaly intensity")

    # Save the plot as an image
    plt.savefig(f'{args.folder_path}/{args.exp_name}_ch_{args.channel_index}_signal.png', dpi=300, bbox_inches='tight')
    plt.close()



    def std_1D(np_array):
        return np_array

    def std_1D_true(np_array):
        mean = np.mean(np_array)
        std = np.std(np_array)
        return (np_array - mean) / std
    
    # Define the figure and axes
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    #fig.suptitle(f'$\\Delta_X:{args.DeltaX} - h:{args.h_opt}$', fontsize=16)
    Delta_list = []

    # Plot S data
    for ii, DeltaX_h_comb_list_ii in enumerate(args.DeltaX_h_comb_list):
        DeltaX = DeltaX_h_comb_list_ii[0][0]
        axes[0, 0].plot(TDE_np[ii, :, args.IM_list.index('t')], 
                        std_1D(TDE_np[ii, :, args.IM_list.index('S')]), 
                        label=rf"S_$\Delta${DeltaX}",
                        color=plt.cm.viridis(ii / len(args.DeltaX_h_comb_list)), 
                        linewidth = 0.7,
                        alpha=0.5
                        )
        axes[0, 0].set_ylabel("Shannon")
        Delta_list.append(DeltaX)

    for ii, DeltaX_h_comb_list_ii in enumerate(args.DeltaX_h_comb_list):
        DeltaX = DeltaX_h_comb_list_ii[0][0]
        axes[1, 0].plot(TDE_np[ii, :, args.IM_list.index('t')], 
                        std_1D_true(TDE_np[ii, :, args.IM_list.index('F')]), 
                        color=plt.cm.viridis(ii / len(args.DeltaX_h_comb_list)), 
                        label=rf"F_$\Delta${DeltaX}",
                        #color='purple',
                        linewidth = 0.7,
                        alpha=0.5
                        )
        axes[1, 0].set_ylabel(r"Fisher")
    

    # Plot TqX data
    for ii, DeltaX_h_comb_list_ii in enumerate(args.DeltaX_h_comb_list):
        DeltaX = DeltaX_h_comb_list_ii[0][0]
        axes[0, 1].plot(TDE_np[ii, :, args.IM_list.index('t')], 
                        std_1D(TDE_np[ii, :, args.IM_list.index('Tq0.8')]), 
                        color=plt.cm.viridis(ii / len(args.DeltaX_h_comb_list)), 
                        label=rf"Tq0.8_$\Delta${DeltaX}",
                        #color='purple',
                        linewidth = 0.7,
                        alpha=0.5
                        )
        axes[0, 1].set_ylabel(r"Tsallis_$q0.8$")

    # Plot TqX data
    for ii, DeltaX_h_comb_list_ii in enumerate(args.DeltaX_h_comb_list):
        DeltaX = DeltaX_h_comb_list_ii[0][0]
        axes[1, 1].plot(TDE_np[ii, :, args.IM_list.index('t')], 
                        std_1D(TDE_np[ii, :, args.IM_list.index('Tq1.5')]), 
                        color=plt.cm.viridis(ii / len(args.DeltaX_h_comb_list)), 
                        label=rf"Tq1.5_$\Delta${DeltaX}",
                        #color='purple',
                        linewidth = 0.7,
                        alpha=0.5
                        )
        axes[1, 1].set_ylabel(r"Tsallis_$q1.5$")

    # Plot RqX data
    for ii, DeltaX_h_comb_list_ii in enumerate(args.DeltaX_h_comb_list):
        DeltaX = DeltaX_h_comb_list_ii[0][0]
        axes[0, 2].plot(TDE_np[ii, :, args.IM_list.index('t')], 
                        std_1D(TDE_np[ii, :, args.IM_list.index('Rq0.8')]), 
                        color=plt.cm.viridis(ii / len(args.DeltaX_h_comb_list)), 
                        label=rf"Rq0.8_$\Delta${DeltaX}",
                        #color='purple',
                        linewidth = 0.7,
                        alpha=0.5
                        )
        axes[0, 2].set_ylabel(r"Rényi_$\alpha0.8$")

    for ii, DeltaX_h_comb_list_ii in enumerate(args.DeltaX_h_comb_list):
        DeltaX = DeltaX_h_comb_list_ii[0][0]
        axes[1, 2].plot(TDE_np[ii, :, args.IM_list.index('t')], 
                        std_1D(TDE_np[ii, :, args.IM_list.index('Rq1.5')]), 
                        color=plt.cm.viridis(ii / len(args.DeltaX_h_comb_list)), 
                        label=rf"Rq1.5_$\Delta${DeltaX}",
                        #color='purple',
                        linewidth = 0.7,
                        alpha=0.5
                        )
        axes[1, 2].set_ylabel(r"Rényi_$\alpha1.5$")

    # Add a vertical line at args.t0_anomaly
    for ax in axes.flatten():
        #ax.axvline(x=args.t0_anomaly, color='red', linestyle='--', label="Anomaly Time")
        #ax.legend(loc='lower left', fontsize=7)
        ax.grid(True)
        #ax.set_xlabel("Time")
        #ax.set_ylabel("Value")
        ax.set_xlim(min(t_i), max(t_i))
        ax.tick_params(axis='y', labelsize=7)
        #ax.axvline(x=7.5-0.3, color='red', linestyle='--', linewidth=0.75)
        #ax.axvline(x=7.5+0.3, color='red', linestyle='--', linewidth=0.75)
        #ax.axvline(x=3036, color='red', linestyle='--', linewidth=0.75)

    axes[1, 0].set_xlabel("time [s]")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 2].set_xlabel("time [s]")
    plt.subplots_adjust(bottom=0.15)  # You can adjust this value as needed


    # Add colorbar with custom levels at the bottom
    cax = fig.add_axes([0.1, 0.05, 0.8, 0.02])  # [left, bottom, width, height]
    cbar = plt.colorbar(cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=plt.cm.viridis), cax=cax, orientation='horizontal')

    # Specify custom levels and labels for the colorbar
    custom_levels = np.linspace(0, 1, len(Delta_list) + 1)  # Use the populated Delta_list
    cbar.set_ticks(custom_levels[:-1] + 0.5 / len(Delta_list))
    cbar.set_ticklabels([rf"{DeltaX}" for DeltaX in Delta_list])
    cbar.set_label(r"$\Delta$", rotation=0, labelpad=15)
    
    # Save the plot as an image
    plt.savefig(f'{args.folder_path}/{args.exp_name}_ch_{args.channel_index}_IM_time_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()



def df_JSD_matrix(args):
    
    def df_plot(df, savepath, highlight_cells=[]):
        # Set the figure size
        plt.figure(figsize=(10, 8))

        sns.heatmap(df, annot=True, fmt=".4f", cmap="viridis")
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    if hasattr(args, "quality_score_df"):
        df_plot(args.quality_score_df, f'{args.folder_path}/{args.exp_name}_ch_{args.channel_index}_JSD_quality_score.png', highlight_cells=args.DeltaX_h_comb_list_JSD)


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
                axes[i, j].plot(x, pdf(x), color=line_color, alpha=1/np.log2(len(pdf_list))*0.15)

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
    plt.savefig(f'{args.folder_path}/{args.exp_name}_ch_{args.channel_index}_pdf_comparative_grid.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_DeltaX_h_quality_score_2D_surface(DeltaX_list, h_list, args):

    # Create a meshgrid from the axes
    H, DeltaX = np.meshgrid(h_list, DeltaX_list)

    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D surface
    ax.plot_surface(H, DeltaX, args.quality_score_df, cmap='viridis')

    # Set labels for the axes
    ax.set_xlabel(r'$h$')
    ax.set_ylabel(r'$\Delta$')
    ax.set_zlabel(r'$JSD$')

    # Show the plot
    plt.savefig(f'{args.folder_path}/{args.exp_name}_ch_{args.channel_index}_2D_JSD_surface.png', dpi=300, bbox_inches='tight')
    plt.close()

def report_results(TDE_np, pdfs_list, supports_list, window_labels_list, dataset, args):

    #Create a folder where save all the results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.folder_path = f'{args.result_report_path}/{args.exp_name}_{timestamp}'
    if not os.path.exists(args.folder_path):
        os.makedirs(args.folder_path)

    #Save experiment settings inside args
    with open(f'{args.folder_path}/{args.exp_name}_ch_{args.channel_index}.txt', 'w') as file:
        for key, value in vars(args).items():
            if len(f" -->{key}\n{value}\n")<1000:
                file.write(f" -->{key}\n{value}\n")

    #PDF best plot
    pdf_anomaly_plot(pdfs_list, supports_list, window_labels_list, args)

    #IM df plots
    IM_time_plots(TDE_np, dataset, args)