% ToA Estimation Performance Analysis Script
% Written by Danny Sinder
% Analyzes FFT and MUSIC algorithm performance using fixed CFR generation
clear; clc;

% Set random seed for reproducibility
rng(42);

% Parameters
Fs = 40e-3;        % Sampling rate in GHz
N = 128;           % FFT size
pilot_index = 58;  % Max pilot index
tau_rms = 75;    % Mean delay factor (in nsec)
up_sample = 8;     % Up sampling factor
num_samples = 10000; % Samples per SNR value
test_plot = false; % True if we want to display plots for testing

% Create table for storing 10 examples
example_table = table('Size', [10, 3], ...
    'VariableTypes', {'double', 'double', 'double'}, ...
    'VariableNames', {'True_ToA', 'FFT_ToA', 'MUSIC_ToA'});

% Parameters for MUSIC algorithm
L = 8;  % Number of expected taps (matching fixed CFR generation)
delay_step = 2.5;
delay_range = (0:delay_step:200)';

% SNR values to test
SNR_values = 30:-3:10;

% Initialize results arrays
fft_errors = zeros(length(SNR_values), 1);
music_errors = zeros(length(SNR_values), 1);

% Initialize waitbar
h = waitbar(0, 'Processing data...');

% Process each SNR value
for snr_idx = 1:length(SNR_values)
    SNR = SNR_values(snr_idx);
    
    % Update waitbar
    waitbar(snr_idx/length(SNR_values), h, ...
        sprintf('Processing SNR=%d dB... (%d/%d)', ...
        SNR, snr_idx, length(SNR_values)));
    
    % Initialize arrays for errors
    fft_all_errors = zeros(num_samples, 1);
    music_all_errors = zeros(num_samples, 1);
    
    % Generate and process samples
    for sample_idx = 1:num_samples
        % Generate CFR using fixed CFR generation
        [CFR_for_FFT, CFR_for_MUSIC, tau_l, h_taps, ~] = generate_fixed_cfr(Fs, N, pilot_index, up_sample, SNR);
        tau_0 = tau_l(1);

        % Extract the relevant part of CFR for processing
        
        % Process with FFT algorithm
        fft_toa = fft_algorithm(CFR_for_FFT);
        error_fft = abs(fft_toa - tau_0);
        fft_all_errors(sample_idx) = error_fft;
        
        % Process with MUSIC algorithm
        music_toa = music_algorithm(CFR_for_MUSIC, L, delay_range, 1/Fs, N);
        if ~isnan(music_toa)
            error_music = abs(music_toa - tau_0);
            music_all_errors(sample_idx) = error_music;
        else
            music_all_errors(sample_idx) = NaN;
        end

        if sample_idx == 1 && test_plot
            % Calculate time axis for plotting
            time_axis_fft = (0:(length(CFR_for_FFT)-1))/8*128/117*25; % Convert to ns
            time_axis_r_h = (0:length(CFR_for_MUSIC)-1) * (25);
    
            figure
            hold on;
            plot(time_axis_r_h, abs(ifft(CFR_for_MUSIC)));
            plot(time_axis_fft, 8 * abs(ifft(CFR_for_FFT)));
            xline(tau_0, 'b--', 'LineWidth', 1.5);
            xline(fft_toa, 'r--', 'LineWidth', 1.5);
            xline(music_toa, 'g--', 'LineWidth', 1.5);
            xlabel('Time (nsec)');
            ylabel('abs(CIR)')
            legend('IFFT(r_h)', 'IFFT(CFR_for_FFT_interp)', 'ToA', 'estimated ToA', 'MUSIC ToA')

            fprintf('Estimated ToA: %.2f nsec\n', fft_toa);
            fprintf('True first tap location (ToA): %.2f nsec\n', tau_0);
            fprintf('MUSIC ToA: %.2f nsec\n', music_toa);
        end

        % table of examples (for testing)
        if sample_idx < 11
            % Store in table
            example_table(sample_idx,:) = {tau_0, fft_toa, music_toa};
        end
        if sample_idx == 11
            % Display example table
            disp('Example ToA Estimations (in ns):');
            disp(example_table);
        end
    end
    
    % Calculate 90th percentile of errors
    fft_errors(snr_idx) = prctile(fft_all_errors, 90);
    valid_music_errors = music_all_errors(~isnan(music_all_errors));
    if ~isempty(valid_music_errors)
        music_errors(snr_idx) = prctile(valid_music_errors, 90);
    else
        music_errors(snr_idx) = NaN;
    end
end

% Close waitbar
close(h);

% Create results table
results_table = table(SNR_values', fft_errors, music_errors, ...
    'VariableNames', {'SNR_dB', 'FFT_90th_Percentile_Error', 'MUSIC_90th_Percentile_Error'});

% Display results
disp('90th Percentile of Absolute ToA Error (nsec) for each algorithm:');
disp(results_table);

% Save results to CSV file
writetable(results_table, 'FFT_MUSIC_error_results.csv');