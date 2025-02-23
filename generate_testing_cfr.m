% CFR Generation Function for testing - Path Delay Estimation project
% Written by Danny Sinder
function [r, r_h, lambda, tau_0 ,tau_l, E_hl_2, sigma_l_2, h, F, sigma_w_2, w] = ...
    generate_testing_cfr(Fs, N, pilot_index, tau_rms, up_sample, L, snr)
    % Simulates the CFR signal based on input parameters with chosen L+SNR
    % Inputs:
    %   Fs          - Sampling rate (in GHz)
    %   N           - FFT size
    %   pilot_index - Max limit of indices for pilot tones
    %   tau_rms     - RMS delay spread (in nsec)
    %   snr_case    - 'l' for low SNR case, 'h' for high SNR case
    %   up_sample   - Up-sampling factor in frequency 
    %   L           - Number of taps
    %   snr         - Signal-to-noise ratio used
    % Outputs:
    %   r           - Simulated noisy Low-res. CFR signal
    %   r_h         - Simulated noiseless High-res. CFR signal
    %   lambda      - Arrival rate
    %   tau_l       - Tap delays
    %   tau_0       - True ToA
    %   E_hl_2      - Expected power of tap
    %   sigma_l_2   - Tap variance
    %   sigma_w_2   - Noise variance
    %   h           - Channel coefficients
    %   F           - Matrix for CFR generation
    %   w           - Noise vector

    % Sampling time
    Ts = 1/Fs; % in nsec
    % Indices for pilot tones
    pilot_indexes_low = -pilot_index:pilot_index;
    pilot_indexes_high = -up_sample*pilot_index:up_sample*pilot_index;
    % Number of pilot signals
    num_pilots = length(pilot_indexes_low);

    %%%%%%%%%%% CFR simulation process %%%%%%%%%%%

    % Step 1: Uniformly pick λ
    lambda = pick_lambda();

    % Step 2: Generate L tap delays τ_l using Poisson process
    [tau_0, tau_l] = generate_tap_delays(L, lambda);

    % Step 3: Calculate expected power of tap E[|h_l|^2] and tap variance σ_l^2
    [E_hl_2, sigma_l_2] = calculate_tap_power(tau_l, tau_rms);

    % Step 4: Generate channel coefficients h_l
    hl = generate_channel_coefficients(sigma_l_2);

    % Step 5: Scale channel coefficients to normalize power
    hl_s = scale_channel_coefficients(hl, E_hl_2);

    % Step 6: Generate h vector
    h = hl_s(:);

    % Step 7: Generate Matrix F & Matrix F_high
    F = generate_matrix_F(N*up_sample, pilot_indexes_high, tau_l, Ts/up_sample);

    % Step 8: Calculate noise frequency variance σ_w^2
    sigma_w_2 = calculate_noise_variance(snr, N*up_sample);

    % Step 9 & 10: Generate noise vector w
    w = generate_noise_vector(sigma_w_2, num_pilots);

    % Step 11: Generate High-res. noiseless CFR signal
    r_h = F * h;

    % Step 12: Generate zero-padded Low-res. noisy CFR signal
    r = r_h((up_sample*pilot_index)+1-pilot_index:(up_sample*pilot_index)+1+pilot_index);
    r = r + w;
    r = [zeros((up_sample-1)*pilot_index,1);r;zeros((up_sample-1)*pilot_index,1)];

end

%%%%%%%%%%% Helper-functions %%%%%%%%%%%
function lambda = pick_lambda()
    lambda = (5 + (45)*rand); % Arrival rate (random within [5, 50]nsec)
end

function [tau_0, tau_l] = generate_tap_delays(L, time_interval) % new option
    lambda = L / time_interval;
    tau = zeros(1,L);
    for k = 2:L
        U = 1.0 -rand(1);
        next_time = -log(U)/lambda;
        tau(k) = tau(k-1) + next_time;
    end
    
    % Add initial delay tau_0
    tau_0 = 50 + (100 * rand);  % ToA tau_0 (random within [50, 150]nsec)
    tau_l = tau + tau_0;  % Sort to maintain increasing order
end

function [E_hl_2, sigma_l_2] = calculate_tap_power(tau_l, tau_rms)
    E_hl_2 = exp(-tau_l/tau_rms);
    sigma_l_2 = E_hl_2 / 2;
end

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
