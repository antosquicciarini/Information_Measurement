#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antoniosquicciarini
KAI - ZEN
"""
import numpy as np

def extract_window(dataset, deltaX, DeltaX, ii):
    data = dataset[int(deltaX*ii) : int(DeltaX+deltaX*ii)]
    data = (data-np.mean(data))/np.std(data) #standardize data
    #data = (data-np.min(data))/(np.max(data)-np.min(data)) #standardize data
    return data

def count_N_windows(N_points, DeltaX, deltaX):
    return int((N_points - DeltaX)//deltaX)