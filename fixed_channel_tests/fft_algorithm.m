% FFT Algorithm Implementation - Path Delay Estimation project
% Written by Danny Sinder
function [estimated_tap_location] = fft_algorithm(input_vector)
    % FFT Algorithm for Tap Estimation Function
    % Input:
    %   input_vector: Input signal vector of length 1024 (N*up_sample)
    %
    % Output:
    %   estimated_tap_location: Estimated location of the first tap


    % Parameters
    time_resolution = 25/8;  % Time resolution in nsec
    threshold_db = 13;
    
     % Step 1: Perform IFFT and get power delay profile
    CIR_1024 = abs(ifft(input_vector)).^2;
    
    % Step 2: Find maximum value and apply threshold
    [max_val, ~] = max(CIR_1024);
    threshold = max_val * 10^(-threshold_db/20);  % Convert dB to linear scale
    
    % Step 5: Find points above threshold
    above_threshold = CIR_1024 >= threshold;
    
    % Step 6: Find the first significant tap (first tap is the strongest)
    valid_indices = find(above_threshold);
    [~, max_idx] = max(CIR_1024(valid_indices));
    first_tap_idx = valid_indices(max_idx);
    
    % Step 7: Check for consecutive taps and apply spline interpolation if needed
    if first_tap_idx > 1 && first_tap_idx < length(CIR_1024)
        % Create finer time points around the first tap
        x_orig = (first_tap_idx-1:first_tap_idx+1)';
        y_orig = CIR_1024(first_tap_idx-1:first_tap_idx+1);
        x_interp = linspace(first_tap_idx-1, first_tap_idx+1, 100)';
        
        % Perform spline interpolation
        y_interp = spline(x_orig, y_orig, x_interp);
        
        % Find maximum in interpolated data
        [~, max_idx] = max(y_interp);
        fine_idx = x_interp(max_idx);
    else
        fine_idx = first_tap_idx;
    end
    
    % Convert index to time
    estimated_tap_location = (fine_idx-1)/8*128/117*25;
end