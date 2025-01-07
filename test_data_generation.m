% Test data generation script - Path Delay Estimation Project
% Written by Danny Sinder
% Run this script to generate two test data files (per SNR case)

clc, clear;

%%%%%%%%%%% General parameters %%%%%%%%%%%

% Folder name where files will be saved
save_folder = 'data'; 

% Number of samples (channels) to generate inside data
num_samples = 200000;

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

% Low SNR case training data generation
filename_low = append(save_folder,'/test_data_high.h5');
generate_data(num_samples, Fs, N, ...
    pilot_index, tau_rms, 'l', up_sample, filename_low);

% High SNR case training data generation
filename_high = append(save_folder,'/test_data_low.h5');
generate_data(num_samples, Fs, N, ...
    pilot_index, tau_rms, 'h', up_sample, filename_high);
