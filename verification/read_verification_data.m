% Read Verification Data Function for Verification - Path Delay Estimation project
% Written by Danny Sinder
function [r, w, h, L, lambda, tau_0, tau_l, E_hl_2, sigma_l_2, sigma_w_2, snr]...
    = read_verification_data(filename)
    % Read HDF5 file and extracts the needed parameters for verification
    % Inputs:
    %   filename    - Input HDF5 filename
    % Outputs:
    %   r           - Simulated CFR signal
    %   w           - Noise vector
    %   h           - Channel coefficients
    %   L           - Number of taps
    %   lambda      - Arrival rate
    %   tau_0       - True ToA
    %   tau_l       - Tap delays
    %   E_hl_2      - Expected power of tap
    %   sigma_l_2   - Tap variance
    %   sigma_w_2   - Noise variance
    %   snr         - Signal-to-noise ratio used

    % Read complex data
    r_real = h5read(filename, '/CFRs_Real');
    r_imag = h5read(filename, '/CFRs_Imag');
    r = r_real + 1i * r_imag;

    w_real = h5read(filename, '/W_Real');
    w_imag = h5read(filename, '/W_Imag');
    w = w_real + 1i * w_imag;

    h_real = h5read(filename, '/H_Real');
    h_imag = h5read(filename, '/H_Imag');
    h = h_real + 1i * h_imag;

    % Read other data
    L = h5read(filename, '/L');
    lambda = h5read(filename, '/Lambda');
    tau_0 = h5read(filename, '/Tau_0');
    tau_l = h5read(filename, '/Tau_L');
    E_hl_2 = h5read(filename, '/E_Hl_2');
    sigma_l_2 = h5read(filename, '/Sigma_L_2');
    sigma_w_2 = h5read(filename, '/Sigma_W_2');
    snr = h5read(filename, '/SNR');
end
