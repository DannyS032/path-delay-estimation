% Data Generation for NN training & testing - Path Delay Estimation project
% Written by Danny Sinder
function generate_data(num_samples, Fs, N, pilot_index, tau_rms, snr_case, up_sample, filename)
    % Generates desired number of training data samples for Generative Network training purposes 
    % Inputs:
    %   num_samples - Number of data samples to generate
    %   Fs          - Sampling rate (in GHz)
    %   N           - FFT size
    %   pilot_index - Index for max pilot tones
    %   tau_rms     - RMS delay spread (in nsec)
    %   snr_case    - 'l' for low SNR case, 'h' for high SNR case
    %   up_sample   - Up sampling factor for freq. resolution 
    %   filename    - Output HDF5 filename
    % Outputs:
    %   <filename>.h5 file that stores all the training data, specifically:
    %       cir_low     - Simulated Low-Res. CIR signal
    %       cir_high    - Simulated High-Res. CIR signal
    %       cfr_high    - Simulated noiseless high-res. CFR signal
    %       tau_0       - True ToA
    %       L           - Number of taps
    %       SNR         - Signal-to-noise ratio used

    % Check if the file already exists and delete it
    if isfile(filename)
        delete(filename);
    end
    
    % Initialize arrays for storing generated data
    cir_low = zeros(num_samples, 2, N*up_sample); 
    cir_high = zeros(num_samples, 2, N*up_sample); 
    cfr_high = zeros(num_samples, 2, N*up_sample); 
    toa = zeros(num_samples, 1); 
    num_taps = zeros(num_samples, 1);
    SNR_db = zeros(num_samples, 1);

    % Initialize the waitbar
    hWaitbar = waitbar(0, 'Generating Training Data...');

    % Data generation and saving
    for i = 1:num_samples
        % Generate the CFRs
        [cfr_l, cfr_h, L, ~, tau_0 , ~, ~, ~, ~, ~, ~, ~, snr] = ...
            generate_cfr(Fs, N, pilot_index, tau_rms, snr_case, up_sample);

        % Generate the CIRs
        cir_l = sqrt(N*up_sample) .* ifft(cfr_l, N*up_sample);
        cir_h = sqrt(N*up_sample) .* ifft(cfr_h, N*up_sample);

        % Zero-pad high res CFR to N*up_sample
        cfr_h = [cfr_h; zeros(N*up_sample-length(cfr_h),1)]; 

        % Store the data
        cir_low(i, 1, :) = real(cir_l); 
        cir_low(i, 2, :) = imag(cir_l);

        cir_high(i, 1, :) = real(cir_h); 
        cir_high(i, 2, :) = imag(cir_h);

        cfr_high(i, 1, :) = real(cfr_h);
        cfr_high(i, 2, :) = imag(cfr_h);

        toa(i) = tau_0;

        num_taps(i) = L;

        SNR_db(i) = snr;

        % Update the waitbar
        progress = i / num_samples;
        waitbar(progress, hWaitbar, sprintf('Generating Data... %d%%', round(progress * 100)));
    end

    % Close the waitbar
    close(hWaitbar);

    % Create and write data to HDF5 file

    h5create(filename, '/CIR_Low', size(cir_low));
    h5write(filename, '/CIR_Low', cir_low);

    h5create(filename, '/CIR_High', size(cir_high));
    h5write(filename, '/CIR_High', cir_high);

    h5create(filename, '/CFR_High', size(cfr_high));
    h5write(filename, '/CFR_High', cfr_high);

    h5create(filename, '/ToA', size(toa));
    h5write(filename, '/ToA', toa);

    h5create(filename, '/L', size(num_taps));
    h5write(filename, '/L', num_taps);

    h5create(filename, '/SNR', size(SNR_db));
    h5write(filename, '/SNR', SNR_db);

end
