% CFR Simulation Generation for Verification - Path Delay Estimation Project
% Written by Danny Sinder
% Running this script will generate 200,000 CFR samples for low and high SNR cases
% (400,000 in total) and saves all the data to HDF5 files
clear, clc
addpath(fileparts(pwd));

% Parameters
Fs = 40e-3; % Sampling rate in GHz
N = 128; % FFT size
pilot_index = 58;
tau_rms = 75; % in nsec
num_samples = 200000; % Number of CFRs to generate

generate_verification_data(Fs, N, pilot_index, tau_rms, num_samples,...
    'l', 'cfr_simulations_low.h5');
generate_verification_data(Fs, N, pilot_index, tau_rms, num_samples,...
    'h', 'cfr_simulations_high.h5');



