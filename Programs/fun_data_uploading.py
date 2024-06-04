#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: antoniosquicciarini

14 March 2022

KAI ZEN
"""
import pyedflib

#%% GENERAL SETTINGS
# Packages to import
import os
import numpy as np
import matplotlib.pyplot as plt
from fun_EEG_metadata_extraction import EEG_metadata_extraction
from scipy.stats import bernoulli


def generate_mixture_distribution(means, stds, weights, size):

    # Validate input sizes
    assert len(means) == len(stds) == len(weights), "Input sizes must be the same"
    
    # Generate samples from each Gaussian distribution
    samples = [np.random.normal(loc=mean, scale=std, size=int(weight * size)) for mean, std, weight in zip(means, stds, weights)]
    
    # Combine samples  to weights
    mixture_samples = np.concatenate(samples)
    
    return mixture_samples


def load_synthetic_dataset(args):

    t = np.linspace(0, args.t_total, int(args.sampling_rate * args.t_total), endpoint=False)
    def gauss_anom(t, t_total, t0_anomaly):
        std = 0.1
        t_mean = (t_total + t0_anomaly) / 2
        anom = 1 / np.sqrt(2 * np.pi * (std ** 2)) * np.exp(-(t - t_mean) ** 2 / (2 * std ** 2))
        return anom

    g_t = np.ones(len(t))
    if args.anomaly_type == "linear":
        mask = t > args.t0_anomaly
        g_t[mask] *= 1 - (t[mask] - args.t0_anomaly) / (args.t_total - args.t0_anomaly)
    elif args.anomaly_type == "gaussian":
        g_t *= 1 - gauss_anom(t, args.t_total, args.t0_anomaly) / np.max(gauss_anom(t, args.t_total, args.t0_anomaly))

    # Initialize the composite signal
    signal = np.zeros(len(t))
    #rand_phase_shift = np.random.uniform(0, 2*np.pi, len(args.freqs))
    rand_phase_shift =  np.array([0.97568103, 1.75374852, 0.8760489])  
    for (amp, freq, phase_shift) in zip(args.amps, args.freqs, rand_phase_shift):
        signal += g_t*amp*np.real(np.exp(1j * (2 * np.pi * freq * t + phase_shift)))

    # Keep the amplitudes in the same places
    #rand_phase_shift_anomaly =  np.append(rand_phase_shift, np.random.uniform(0, 2*np.pi, len(args.freqs_anom)-len(args.freqs)))
    rand_phase_shift_anomaly =  np.append(rand_phase_shift, np.array([0.95102242, 3.17437698]))

    for (amp, freq, phase_shift) in zip(args.amps_anom, args.freqs_anom, rand_phase_shift_anomaly):
        signal += (1.0-g_t)*amp*np.real(np.exp(1j * (2 * np.pi * freq * t + phase_shift)))
        
    args.phase_shift = rand_phase_shift_anomaly
    
    # Add Gaussian noise
    N_distr = len(args.gaussian_noise_mean)
    noise_samples = np.array([])
    for ii in range(N_distr):
        samples_i = np.random.normal(args.gaussian_noise_mean[ii], args.gaussian_noise_std[ii], int(np.ceil(len(t)/N_distr)))
        noise_samples = np.concatenate([noise_samples, samples_i])
    np.random.shuffle(noise_samples)

    N_distr = len(args.gaussian_noise_mean_anom)
    noise_samples_anom = np.array([])
    for ii in range(N_distr):
        samples_i = np.random.normal(args.gaussian_noise_mean_anom[ii], args.gaussian_noise_std_anom[ii], int(np.ceil(len(t)/N_distr)))
        noise_samples_anom = np.concatenate([noise_samples_anom, samples_i])
    np.random.shuffle(noise_samples_anom)
    
    # Vectorized generation of random samples
    flags = bernoulli(g_t).rvs(len(signal))
    signal += flags * noise_samples[:len(signal)] + (1 - flags) * noise_samples_anom[:len(signal)]

    # trasform 1D to 2D tensor
    signal = signal[:, np.newaxis]

    return signal, args


def load_CHBM_dataset(args):

    dataset = []

    chb = 'chb'+str(args.chb_indx).zfill(2)

    with open(f'{args.path_EEG_database}/{chb}/{chb}-summary.txt', "r") as file:
        # Read the entire file content
        chb_txt_info = file.read()

    EEG_metadata_extraction(chb_txt_info, args)

    edf_file = pyedflib.EdfReader(f'{args.path_EEG_database}/{chb}/{args.chb_path}')

    num_signals = edf_file.signals_in_file

    args.signal_labels = edf_file.getSignalLabels()

    # Read the data from the EDF file
    for i in range(num_signals):
        signal_data = edf_file.readSignal(i)
        dataset.append(signal_data)
    dataset = np.array(dataset).T

    #Without anomaly time reference
    args.t0_anomaly = dataset.shape[0] /args.sampling_rate
    args.t_total = dataset.shape[0] /args.sampling_rate

    return dataset, args


def data_uploading(args, indx): 
    
    if args.dataset == "synthetic":
        dataset = load_synthetic_dataset(args)
    elif args.dataset == "EEG":
        dataset = load_CHBM_dataset(args)

    return dataset