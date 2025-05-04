import numpy as np
from numpy import linalg as LA
import os
import glob
from pathlib import Path
import pickle
import pandas as pd
from collections import defaultdict

def lazy_random_walk(adj):
    d = adj.sum(0)
    P_t = adj/d
    P_t[np.isnan(P_t)] = 0
    P = 0.5 * (np.identity(P_t.shape[0]) + P_t)
    return P

#Manual scattering. Not learnable. Talk to Dhanajay about NeuroGSTH.
def graph_wavelet(P):
    psi = []
    scales = [1, 2, 4, 8, 16]
    for d1 in scales:
        W_d1 = LA.matrix_power(P, d1) - LA.matrix_power(P, 2 * d1)
        psi.append(W_d1)
    return psi

def zero_order_feature(A, ro):
    return np.matmul(LA.matrix_power(A, 16), ro)

def first_order_feature(A, u, wavelet_outputs):
    power_A = LA.matrix_power(A, 16)
    F1_list = []
    for wavelet_out in wavelet_outputs:
        F1_list.append(np.matmul(power_A, wavelet_out))
    
    F1 = np.concatenate(F1_list, axis=0)
    return F1


def second_order_feature(A, W, u, wavelet_outputs):
    power_A = LA.matrix_power(A, 16)
    F2_list = []
    for i in range(1, len(W)):
        for j in range(i):
            u1 = np.matmul(W[i], wavelet_outputs[j])
            u1 = np.abs(u1)
            F2_list.append(np.matmul(power_A, u1))
    
    F2 = np.concatenate(F2_list, axis=0)
    return F2

def generate_timepoint_feature(adj, ro):
    P = lazy_random_walk(adj)
    W = graph_wavelet(P)
    wavelet_outputs = []
    for w in W:
        wavelet_out = np.matmul(w, ro)
        wavelet_outputs.append(np.abs(wavelet_out))

    F0 = zero_order_feature(P, ro) # EXPECT(48 x 7850)
    F1 = first_order_feature(P, ro, wavelet_outputs) #EXPECT (240 x 7850)
    F2 = second_order_feature(P, W, ro, wavelet_outputs) #(EXPECT480 x 7850)
    F = np.concatenate([F0, F1, F2], axis=0) # EXPECT (768 x 7850)
    return F

def gaussian_kernel(x, y, sigma=20):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

def construct_adjacency_matrix(channel_coordinates, threshold):
    n_points = len(channel_coordinates)
    adj_matrix = np.zeros((n_points, n_points))
    
    #Go through all channle pairs and calc weights
    for i in range(n_points):
        for j in range(i + 1, n_points):
            weight = gaussian_kernel(channel_coordinates[i], channel_coordinates[j])
            if weight > threshold:
                adj_matrix[i, j] = weight
                adj_matrix[j, i] = weight
    return adj_matrix

def create_task_labels(task_segments):
    n_timepoints = 7850  # hard coded as of now but ask Dhanajay about this
    detailed_labels = np.empty(n_timepoints, dtype='object')
    general_labels = np.empty(n_timepoints, dtype='object')
    timepoints = np.arange(n_timepoints)
    
    for start, end, task_name in task_segments:
        mask = (timepoints >= start) & (timepoints < end)
        detailed_labels[mask] = task_name
        if 'Rest' in task_name:
            general_labels[mask] = 'Rest'
        elif 'Improv' in task_name:
            general_labels[mask] = 'Improv'
        elif 'Scale' in task_name:
            general_labels[mask] = 'Scale'
        else:
            general_labels[mask] = 'Other'
    return detailed_labels, general_labels, timepoints

def load_and_reshape_scattering_coeffs(scattering_dir):
    scattering_dir = Path(scattering_dir)
    scattering_files = list(scattering_dir.glob('subject_*_scattering_coeffs.csv'))
    if not scattering_files:
        raise ValueError(f"No files in {scattering_dir} - check path again")
    first_file = np.loadtxt(scattering_files[0], delimiter=',')
    num_timepoints, num_features = first_file.shape
    sample_data = first_file.T
    F0 = sample_data[:48]  
    F1 = sample_data[48:48+48*5]  
    F2 = sample_data[48+48*5:] 
    print(f"F0: {F0.shape}, F1: {F1.shape}, F2: {F2.shape}")
    
    reshaped = reshape_features(F0, F1, F2, sample_data)
    return {
        'F0': F0,
        'F1': F1,
        'F2': F2,
        'F': sample_data,
        'reshaped': reshaped
    }

def reshape_features(F0, F1, F2, F):
    num_nodes = 48 
    num_timepoints = F.shape[1]
    features_per_node = 16
    reshaped = np.zeros((num_nodes, features_per_node, num_timepoints))
    reshaped[:, 0, :] = F0 

    for i in range(num_nodes):
        reshaped[i, 1:6, :] = F1[i*5:(i+1)*5, :]  
        reshaped[i, 6:16, :] = F2[i*10:(i+1)*10, :] 
    
    #Check if anything got messed up
    flattened = reshaped.reshape(-1, num_timepoints)
    error = np.mean(np.abs(flattened - F))
    print(f"Reshaped to {reshaped.shape}, error: {error:.6f}")
    
    return reshaped