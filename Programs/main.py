#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antoniosquicciarini
KAI ZEN
"""

#%% GENERAL SETTINGS
# Packages to import
from fun_pdf_generation import pdf_generation
from fun_data_uploading import data_uploading
from fun_IM_generation import TDE_computation, stat_momentum_calculator
from fun_result_report import report_results
import sys
import json
from fun_parameter_setting import parameter_setting
import itertools
from itertools import product
import time
import numpy as np
import os
from pathlib import Path

import subprocess
if 'darwin' in sys.platform:
    print('Running \'caffeinate\' on MacOSX to prevent the system from sleeping')
    subprocess.Popen('caffeinate')

def set_project_root():
    """Set the current working directory to the project's root."""
    if "Programs" in Path.cwd().parts:
        os.chdir("..")

# paths offline
paths = {
    "path_EEG_database": "Data/raw_datasets/chb-mit-scalp-eeg-database-1.0.0",
    "obj_EEG_dir": "Data/samples/EEG_Data/EEG_IM_objs",
    "result_report_path": "Results/IM_time_signal_analysis/hDelta_JSD_opt",

    "chdir": "/Users/antoniosquicciarini/ownCloud/PhD_Projects/Information_Measurement",
    "add_dir": "/Users/antoniosquicciarini/ownCloud/PhD_Projects/General_Functions"
}

def load_parameters_from_json(json_filename):
    with open(json_filename, 'r') as json_file:
        params = json.load(json_file)
    return params

def generate_settings_combinations(original_dict):
    # Create a list of keys with lists as values
    list_keys = [key for key, value in original_dict.items() if isinstance(value, list)]
    # Generate all possible combinations
    combinations = list(itertools.product(*[original_dict[key] for key in list_keys]))
    # Create a list of dictionaries with unique combinations
    result = []

    for combo in combinations:
        new_dict = original_dict.copy()
        for key, value in zip(list_keys, combo):
            new_dict[key] = value
        result.append(new_dict)

    return result

def calculate_elapsed_time(start_time, ii, params_list):
    # Record the end time
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    # Calculate days, hours, minutes, and seconds
    days = elapsed_time // 86400  # 86400 seconds in a day
    hours = (elapsed_time % 86400) // 3600  # 3600 seconds in an hour
    minutes = (elapsed_time % 3600) // 60  # 60 seconds in a minute
    seconds = elapsed_time % 60
    print(f"### To complete the simulation last: {days} days, {hours} hours, {minutes} minutes, {seconds:.2f} seconds ")

    remaining_time = elapsed_time * (len(params_list) - ii - 1)
    # Calculate days, hours, minutes, and seconds
    days = remaining_time // 86400  # 86400 seconds in a day
    hours = (remaining_time % 86400) // 3600  # 3600 seconds in an hour
    minutes = (remaining_time % 3600) // 60  # 60 seconds in a minute
    seconds = remaining_time % 60
    print(f"### Remaining time: {days} days, {hours} hours, {minutes} minutes, {seconds:.2f} seconds ")


def main():
    print("Start!")
    set_project_root()

    if len(sys.argv) > 1:
        code_cloud = True
        json_name = sys.argv[1]
        print(f"JSON name: {json_name}")

    else:
        code_cloud = False
        json_name = 'KDE_EEG_chb1_01' 
        
    json_filename = f'Programs/JSON_settings/{json_name}.json' 
    params_json = load_parameters_from_json(json_filename)
    params_list = generate_settings_combinations(params_json)

    #Simulation iteration
    for indx, params in enumerate(params_list):

        start_time = time.time()
        print(f"starting time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")     
        print(f"computing {indx} simulation")
        args = parameter_setting(params, paths, json_name, code_cloud)
        dataset_multichannel, args = data_uploading(args, indx)  

        #Iteration over the channels
        for ii in range(dataset_multichannel.shape[1]):
        
            args.channel_index = ii
            dataset = dataset_multichannel[:, args.channel_index]
            pdfs_list, supports_list, window_labels_list = pdf_generation(dataset, args, indx)  
            TDE_np = TDE_computation(pdfs_list, supports_list, window_labels_list, args, indx) 
            report_results(TDE_np, pdfs_list, supports_list, window_labels_list, dataset, args)
            calculate_elapsed_time(start_time, indx, params_list)

if __name__ == "__main__":
    main()

