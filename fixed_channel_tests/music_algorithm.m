% MUSIC Algorithm Implementation - Path Delay Estimation project
% Written by Danny Sinder
function [tap_location] = music_algorithm(input_vector, L, delay_range, Ts, N_FFT)
    % MUSIC Algorithm Implementation for Channel Tap Estimation
    % Inputs:
    %   input_vector: Input signal vector of length 117
    %   L: Number of expected taps
    %   delay_range: Vector of possible delay values to search
    %
    % Output:
    %   tap_location: Estimated location of the first tap

% Step 1: Build the data matrix using sliding window
N = 59; % Window size
A = zeros(N, N);
for i = 1:N
    A(:,i) = input_vector(i:i+N-1);
end

% Step 2: Perform SVD
[U, ~, ~] = svd(A);

% Step 3: Get signal subspace (Es)
Es = U(:, 1:L);

% Step 4: Define noise subspace projection matrix
noise_proj = eye(N) - Es*Es';

% Step 5: Generate search vectors and perform MUSIC spectrum calculation
P_music = zeros(size(delay_range));
for i = 1:length(delay_range)
    tau = delay_range(i) / Ts;
    % Create steering vector
    a = exp(-1j*2*pi*(0:N-1)'*tau/N_FFT);
    % Calculate MUSIC spectrum
    P_music(i) = -10 * log10(real(a'*noise_proj*a));
end

% Step 6: Find peaks in the MUSIC spectrum and apply 17 dB threshold
[peak_values, peak_locs] = findpeaks(abs(P_music));
max_peak = max(P_music);
threshold = max_peak - 17;  % 17 dB below maximum peak

% Find peaks that meet the threshold
valid_peaks_mask = peak_values >= threshold;
valid_peak_locs = peak_locs(valid_peaks_mask);

% Step 7: Get the first tap location (earliest in time)
if ~isempty(valid_peak_locs)
    tap_location = delay_range(min(valid_peak_locs));
else
    tap_location = NaN; 

end