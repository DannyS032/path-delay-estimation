% ToA Error Analysis Script with Online CFR Generation
% Analyzes MUSIC algorithm performance by generating CFRs
% Written by Danny Sinder

clear; clc;

% Set random seed to match original data generation
rng(42);

% Parameters (matching the test data generation)
Fs = 40e-3;      % Sampling rate in GHz
N = 128;         % FFT size
pilot_index = 58;  % Max pilot index
tau_rms = 75;    % Mean delay factor (in nsec)
up_sample = 2;   % Up sampling factor
num_samples = 10000;  % Total samples per L-SNR combination

% Create delay range vector for MUSIC algorithm (0 to 200 nsec with 2.5 nsec resolution)
delay_step = 2.5;
delay_range = (0:delay_step:200)';

% Initialize arrays for L and SNR values (matching generate_test_data.m order)
L_values = 3:15;
SNR_values = 30:-3:-5;

% Initialize results table
results_table = zeros(length(SNR_values), length(L_values));
fd_rates = zeros(length(SNR_values), 1);  % New array for FD rates
fd_counts = zeros(length(SNR_values), 1);  % To track total samples per SNR
fd_failures = zeros(length(SNR_values), 1);  % To track failures per SNR

% Initialize waitbar
h = waitbar(0, 'Processing data...');
total_combinations = length(L_values) * length(SNR_values);
current_combination = 0;

% Process each L-SNR combination
for l_idx = 1:length(L_values)
    L = L_values(l_idx);
    
    for snr_idx = 1:length(SNR_values)
        SNR = SNR_values(snr_idx);
        current_combination = current_combination + 1;
        
        % Update waitbar
        waitbar(current_combination/total_combinations, h, ...
            sprintf('Processing L=%d, SNR=%d dB... (%d/%d)', ...
            L, SNR, current_combination, total_combinations));
        
        % Initialize array for errors for this L-SNR combination
        all_errors = zeros(num_samples, 1);
        
        % Generate and process samples
        for sample_idx = 1:num_samples
            % Generate CFR using the same function as in test data generation
            [cfr_l, ~, ~, tau_0, ~, ~, ~, ~, ~, ~, ~] = ...
                generate_testing_cfr(Fs, N, pilot_index, tau_rms, up_sample, L, SNR);
            
            % Process CFR directly with MUSIC algorithm (extract the relevant part)
            input_vector = cfr_l(pilot_index + 1:end - pilot_index);
            
            % Run MUSIC algorithm
            estimated_toa = music_algorithm(input_vector, L, delay_range, 1/Fs, N);
            
            % Calculate absolute error
            if ~isnan(estimated_toa)
                error = abs(estimated_toa - tau_0);
                all_errors(sample_idx) = error;

                % Update FD statistics
                fd_counts(snr_idx) = fd_counts(snr_idx) + 1;
                if error > 12.5  % 12.5 nsec threshold
                    fd_failures(snr_idx) = fd_failures(snr_idx) + 1;
                end
            else
                all_errors(sample_idx) = NaN;
            end
        end
        
        % Calculate 90th percentile of errors for this L-SNR combination
        valid_errors = all_errors(~isnan(all_errors));
        if ~isempty(valid_errors)
            results_table(snr_idx, l_idx) = prctile(valid_errors, 90);
        else
            results_table(snr_idx, l_idx) = NaN;
        end
    end
end

% Calculate FD rates
fd_rates = fd_failures ./ fd_counts;

% Close waitbar
close(h);

% Create table with row and column labels
L_labels = arrayfun(@(x) sprintf('L=%d', x), L_values, 'UniformOutput', false);
SNR_labels = arrayfun(@(x) sprintf('SNR=%d', x), SNR_values, 'UniformOutput', false);

% Flip the results table to match desired SNR order
results = array2table(results_table, 'VariableNames', L_labels, 'RowNames', SNR_labels);

% Create FD rates table
fd_results = array2table(fd_rates, 'VariableNames', {'FD_Rate'}, 'RowNames', SNR_labels);

% Display results
disp('90th Percentile of Absolute ToA Error (nsec) for each L-SNR combination:');
disp(results);
disp('False Detection Rates per SNR:');
disp(fd_results);

% Save results to MAT file
save('toa_error_analysis_online.mat', 'results', 'L_values', 'SNR_values');

% Save results to CSV file
writetable(results, 'test_results_l_snr/music/MUSIC_final_results.csv', 'WriteRowNames', true);
writetable(fd_results, 'test_results_l_snr/music/MUSIC_fd_rates.csv', 'WriteRowNames', true);

% Create heatmap visualization
figure;
results_matrix = table2array(results);
imagesc(results_matrix);
colorbar;
title('90th Percentile of ToA Error (nsec)');
xlabel('Number of Taps (L)');
ylabel('SNR (dB)');
xticks(1:length(L_values));
xticklabels(L_labels);
yticks(1:length(SNR_values));
yticklabels(SNR_labels);
colormap('jet');
savefig('test_results_l_snr/music/toa_error_heatmap');