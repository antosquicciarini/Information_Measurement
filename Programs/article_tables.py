#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antoniosquicciarini
KAI ZEN
"""

#%% GENERAL SETTINGS
# Packages to import
from fun_pdf_generation import data_generation
from fun_data_analysis import data_analysis
import sys
import numpy as np
import multiprocessing as mp
import os
import pandas as pd

import subprocess
if 'darwin' in sys.platform:
    print('Running \'caffeinate\' on MacOSX to prevent the system from sleeping')
    subprocess.Popen('caffeinate')


#%% DIRECTIONS
# path_database - path where the database folder have been saved
path_database = "/Volumes/T7/EEG_Data/chb-mit-scalp-eeg-database-1.0.0/"
chdir = "/Users/antoniosquicciarini/ownCloud/PhD Projects/Information_Measurement/"
# Append Working directory where have been stored useful function
add_dir = "/Users/antoniosquicciarini/ownCloud/PhD Projects/General_Functions/"
obj_dir = "/Volumes/T7/EEG_Data/EEG_IM_objs/"

os.chdir(chdir)
table_path = "Data_and_Models/ELIO_Article_Files/Tables/"


#%% PANDAS' TABLE

# Setting tables
d = {'$\Delta$': ["2E-13"], '$\delta$': ["2E-9"], '$q$ values': ["$0.7$, $1.5$, $2$, $3$, $4$"]}
df = pd.DataFrame(data=d)
with open( table_path + 'settings.tex', 'w') as tf:

    #os.path.join(os.getcwd(), "manuscript", "src", "tables", "tbl_examples.tex"), "w" 
#) as tf:
    tf.write(
        df
        #.round(1)
        #.reset_index()
        .to_latex(
        index=False,
        #columns=False,
        caption="This is the caption",
        label="tab: table_label",
        escape=False, 
        column_format="ccc",
        )
    )

# Most informative channels



