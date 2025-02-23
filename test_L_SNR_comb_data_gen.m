% Test data generation script - Path Delay Estimation Project
% Written by Danny Sinder
% Run this script to generate file for L+SNR combination testing

clc, clear;

%%%%%%%%%%% General parameters %%%%%%%%%%%

% Folder name where files will be saved
save_folder = 'data'; 

% Number of samples (channels) to generate inside data
num_samples = 10000;

% Sample rate (BW) in GHz
Fs = 40e-3;

% FFT size
N = 128;

% Max pilot index (so the indexes used will be -pilot_index:pilot_index)
pilot_index = 58;

% Mean delay factor (in nsec)
tau_rms = 75;

% Up sample factor for the High-res. CFR
up_sample = 2;

%%%%%%%%%%% Test Data Generation Process %%%%%%%%%%%

% Ensure save folder exists
if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end

% test data generation
generate_test_data(num_samples, Fs, N, pilot_index, tau_rms, up_sample, save_folder, true);
