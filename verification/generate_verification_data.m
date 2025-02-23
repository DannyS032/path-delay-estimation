% CFR Data Generation for Verification Function - Path Delay Estimation project
% Written by Danny Sinder
function generate_verification_data(Fs, N, pilot_index, tau_rms, num_samples, snr_case, filename)
    % Generates desired number of CFR simulations data for verification purposes
    % Inputs:
    %   Fs          - Sampling rate (in GHz)
    %   N           - FFT size
    %   pilot_index - Index for max pilot tones
    %   tau_rms     - RMS delay spread (in nsec)
    %   num_samples - Number of CFRs to generate
    %   snr_case    - 'l' for low SNR case, 'h' for high SNR case
    %   filename    - Output HDF5 filename
    % Outputs:
    %   <filename>.h5 file that stores all the data of the simulated channels 
    %   list of saved data needed for verification purposes:
    %   r           - Simulated CFR signal
    %   L           - Number of taps
    %   lambda      - Arrival rate
    %   tau_l       - Tap delays
    %   tau_0       - True ToA
    %   E_hl_2      - Expected power of tap
    %   sigma_l_2   - Tap variance
    %   sigma_w_2   - Noise variance
    %   h           - Channel coefficients
    %   w           - Noise vector
    %   snr         - Signal-to-noise ratio used

    % Check if the file already exists and delete it
    if isfile(filename)
        delete(filename);
    end

    % Initialize storage arrays
    max_L = 15; % Max value for L 
    max_r_length = 117; % Length of r (from pilot_indexes)
    
    % Initialize arrays for storing generated data
    all_r_real = NaN(max_r_length, num_samples); 
    all_r_imag = NaN(max_r_length, num_samples); 
    all_w_real = NaN(max_r_length, num_samples); 
    all_w_imag = NaN(max_r_length, num_samples);
    all_h_real = NaN(max_L, num_samples); 
    all_h_imag = NaN(max_L, num_samples);
    all_tau_l = NaN(max_L, num_samples);
    all_sigma_l_2 = NaN(max_L, num_samples);
    all_E_hl_2 = NaN(max_L, num_samples);
    all_L = NaN(1, num_samples);
    all_lambda = NaN(1, num_samples);
    all_tau_0 = NaN(1, num_samples);
    all_sigma_w_2 = NaN(1, num_samples);
    all_snr = NaN(1, num_samples);

    % Initialize the waitbar
    hWaitbar = waitbar(0, 'Generating CFRs...');

    % CFR generation and saving
    for i = 1:num_samples
        % Generate the CFR and metadata
        [r, ~, L, lambda, tau_0, tau_l, E_hl_2, sigma_l_2, h, ~, sigma_w_2, w, snr] = ...
            generate_cfr(Fs, N, pilot_index, tau_rms, snr_case, 1);

        % Store the CFR signal with padding (length 117)
        all_r_real(1:length(r), i) = real(r(:));
        all_r_imag(1:length(r), i) = imag(r(:));
        all_w_real(1:length(w), i) = real(w(:));
        all_w_imag(1:length(w), i) = imag(w(:));

        % Store other parameters
        all_L(i) = L;
        all_lambda(i) = lambda;
        all_tau_0(i) = tau_0;
        all_sigma_w_2(i) = sigma_w_2;
        all_snr(i) = snr;

        % Store the channel coefficients, tap delays, tap variances and
        % expected power of taps with padding (length L)
        all_h_real(1:L, i) = real(h(:));
        all_h_imag(1:L, i) = imag(h(:));
        all_tau_l(1:L, i) = tau_l(:);
        all_sigma_l_2(1:L, i) = sigma_l_2(:);
        all_E_hl_2(1:L, i) = E_hl_2(:);

        % Update the waitbar
        progress = i / num_samples;
        waitbar(progress, hWaitbar, sprintf('Generating CFRs... %d%%', round(progress * 100)));
    end

    % Close the waitbar
    close(hWaitbar);

    % Create and write data to HDF5 file for complex numbers using real and imaginary parts

    % Real part of CFRs
    h5create(filename, '/CFRs_Real', size(all_r_real));
    h5write(filename, '/CFRs_Real', all_r_real);

    % Imaginary part of CFRs
    h5create(filename, '/CFRs_Imag', size(all_r_imag));
    h5write(filename, '/CFRs_Imag', all_r_imag);

    % Real part of Noise
    h5create(filename, '/W_Real', size(all_w_real));
    h5write(filename, '/W_Real', all_w_real);

    % Imaginary part of Noise
    h5create(filename, '/W_Imag', size(all_w_imag));
    h5write(filename, '/W_Imag', all_w_imag);

    % Real part of Channel Coefficients
    h5create(filename, '/H_Real', size(all_h_real));
    h5write(filename, '/H_Real', all_h_real);

    % Imaginary part of Channel Coefficients
    h5create(filename, '/H_Imag', size(all_h_imag));
    h5write(filename, '/H_Imag', all_h_imag);

    % Create and write data to HDF5 file for real numbers
    h5create(filename, '/L', size(all_L));
    h5write(filename, '/L', all_L);

    h5create(filename, '/Lambda', size(all_lambda));
    h5write(filename, '/Lambda', all_lambda);
    
    h5create(filename, '/Tau_0', size(all_tau_0));
    h5write(filename, '/Tau_0', all_tau_0);
    
    h5create(filename, '/Tau_L', size(all_tau_l));
    h5write(filename, '/Tau_L', all_tau_l);
    
    h5create(filename, '/E_Hl_2', size(all_E_hl_2));
    h5write(filename, '/E_Hl_2', all_E_hl_2);
    
    h5create(filename, '/Sigma_L_2', size(all_sigma_l_2));
    h5write(filename, '/Sigma_L_2', all_sigma_l_2);
    
    h5create(filename, '/Sigma_W_2', size(all_sigma_w_2));
    h5write(filename, '/Sigma_W_2', all_sigma_w_2);
    
    h5create(filename, '/SNR', size(all_snr));
    h5write(filename, '/SNR', all_snr);
end
