% Fixed CFR Generation Function for testing - Path Delay Estimation project
% Written by Danny Sinder
function [CFR_for_FFT_Interp, r_h, tau_l, h, w] = ...
    generate_fixed_cfr(Fs, N, pilot_index, up_sample, snr)
    % Simulates the CFR signal based on input parameters with chosen L+SNR
    % Inputs:
    %   Fs          - Sampling rate (in GHz)
    %   N           - FFT size
    %   pilot_index - Max limit of indices for pilot tones
    %   tau_rms     - RMS delay spread (in nsec)
    %   up_sample   - Up-sampling factor in frequency 
    %   snr         - Signal-to-noise ratio used
    % Outputs:
    %   r           - Simulated noisy Low-res. CFR signal
    %   r_h         - Simulated noiseless High-res. CFR signal
    %   tau_0       - True ToA
    %   tau_l       - Tap delays
    %   w           - Noise vector

    % Sampling time
    Ts = 1/Fs; % in nsec
    % Indices for pilot tones
    pilot_indexes_low = -pilot_index:pilot_index;
    pilot_indexes_high = -up_sample*pilot_index:up_sample*pilot_index;
    % Number of pilot signals
    num_pilots = length(pilot_indexes_low);

    %%%%%%%%%%% Channel Parameters as specified %%%%%%%%%%%

    % Number of taps
    nTaps = 8;
    
    % Generate tap delays as specified
    delay_taps_1st = 10e-9; % 10ns base delay for first tap
    delay_taps = zeros(1, nTaps);
    delay_taps(1) = delay_taps_1st + 5e-9*2*(rand(1)-0.5); % First tap with ±5ns jitter
    delay_taps(2:end) = 50e-9*rand(1,nTaps-1) + delay_taps(1); % Remaining taps
    delay_taps = delay_taps .* 1e9; % covert to nsec
    
    % Tap gains in dB as specified
    db_taps_pattern = [0 -3 -4 -5 -6 -7 -8 -10];
    db_taps = db_taps_pattern;
    db_taps(2:end) = db_taps(2:end) + 0.5*2*(rand(size(db_taps(2:end)))-0.5);
    
    % Convert dB to linear scale
    linear_taps = 10.^(db_taps/10);

    % Sort delays and reorder gains accordingly
    [sorted_delay_taps, sort_idx] = sort(delay_taps);
    sorted_taps_gain = linear_taps(sort_idx);

    %%%%%%%%%%% CFR simulation process %%%%%%%%%%%

    % Step 1: Generate tau_0 & tau_l from Channel Parameters
    tau_0 = 50 + (100 * rand);  % ToA tau_0 (random within [50, 150]nsec)
    tau_l = sorted_delay_taps + tau_0;

    % Step 2: Calculate expected power of tap E[|h_l|^2] and tap variance σ_l^2
    E_hl_2 = linear_taps;
    sigma_l_2 = E_hl_2 / 2;

    % Step 3: Generate channel coefficients h_l
    hl = generate_channel_coefficients(sigma_l_2);

    % Step 4: Scale channel coefficients to normalize power
    hl_s = scale_channel_coefficients(hl, E_hl_2);

    % Step 5: Generate h vector
    h = hl_s(:);

    % Step 6: Generate Matrix F & Matrix F_high
    F = generate_matrix_F(N, pilot_indexes_low, tau_l, Ts);

    % Step 7: Calculate noise frequency variance σ_w^2
    sigma_w_2 = calculate_noise_variance(snr, N*up_sample);

    % Step 8 & 9: Generate noise vector w
    w = generate_noise_vector(sigma_w_2, num_pilots);

    % Step 10: Generate High-res. noiseless CFR signal
    r_h = F * h;

    % Step 11: Generate zero-padded Low-res. noisy CFR signal
    r = r_h + w;
    
    % For the FFT interploation create vector of 1024 = 128*8
    CFR_for_FFT_Interp = zeros(N*up_sample,1);

    % Insert CFR+Noise to IFFT buffer
    CFR_for_FFT_Interp(1:(pilot_index+1)) = r(1:(pilot_index+1));

    CFR_for_FFT_Interp(N*up_sample-pilot_index+1:N*up_sample) = r((pilot_index+2):num_pilots);

end   

%%%%%%%%%%% Helper-functions %%%%%%%%%%%
function hl = generate_channel_coefficients(sigma_l_2)
    hl = sqrt(sigma_l_2) .* (randn(1, length(sigma_l_2)) + 1i*randn(1, length(sigma_l_2)));
end

function hl_s = scale_channel_coefficients(hl, E_hl_2)
    P_channel = sum(E_hl_2); % Total channel power
    hl_s = hl / sqrt(P_channel); % Scaled channel coefficients
end

function F = generate_matrix_F(N, indexes, tau_l, Ts)
    indexes = indexes(:);  
    tau_l = tau_l(:)';    
    F = exp(kron(indexes, (tau_l * (-1i * 2 * pi / (N * Ts))))) / sqrt(N);
end

function sigma_w_2 = calculate_noise_variance(snr, N)
    sigma_w_2 = (1/N) * 10^(-snr/10);
end

function w = generate_noise_vector(sigma_w_2, num_pilots)
    wn = sqrt(sigma_w_2) * (randn(1, num_pilots) + 1i*randn(1, num_pilots));
    w = wn(:) ./ sqrt(2);
end
