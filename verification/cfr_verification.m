% Verification Script - Path Delay Estimation Project
% Written by Danny Sinder
% Oder of script:
%   General parameters
%   Data loading from files
%   Metric organization & calculation
%   PDF & CDF plotting
%   KS & AD tests
%   Mean, t-test & variance tests
%   Helper-functions
clear; clc;

%%%%%%%%%%% General parameters %%%%%%%%%%%

num_iterations = 100;
num_samples = 100000;
display_results = false; % true - to display the results, false - not display
display_plots = false; % true - to display the plots, false - not display

%%%%%%%%%%% Data loading from files %%%%%%%%%%%

% Define file names for low and high SNR cases
file_low = 'cfr_simulations_low.h5';
file_high = 'cfr_simulations_high.h5';

% Load verification data
[r_low, w_low, h_low, L_low, lambda_low, tau_0_low, tau_l_low, E_hl_2_low,...
    sigma_l_2_low, sigma_w_2_low, snr_low] = read_verification_data(file_low);
[r_high, w_high, h_high, L_high, lambda_high, tau_0_high, tau_l_high, E_hl_2_high,...
    sigma_l_2_high, sigma_w_2_high, snr_high] = read_verification_data(file_high);

%%%%%%%%%%% Metric organization & calculation %%%%%%%%%%%

% Calculate metrics for the low SNR case
metrics_low = calculate_metrics(L_low, lambda_low, tau_0_low, tau_l_low, E_hl_2_low, ...
                                sigma_l_2_low, h_low, w_low, sigma_w_2_low, snr_low);

% Calculate metrics for the high SNR case
metrics_high = calculate_metrics(L_high, lambda_high, tau_0_high, tau_l_high, E_hl_2_high, ...
                                 sigma_l_2_high, h_high, w_high, sigma_w_2_high, snr_high);

%%%%%%%%%%% PDF & CDF plotting %%%%%%%%%%%

plot_cdf_pdf(metrics_high, metrics_low, 'pdf_cdf_plots_verification', display_plots);

%%%%%%%%%%% KS & AD tests %%%%%%%%%%%

% Define thresholds
ks_thresholds = [1.354, 1.617]; % 95% and 99% thresholds for KS test
ad_thresholds = [3.788, 4.882]; % 95% and 99% thresholds for AD test
thresholds_labels = {'95%', '99%'};

% λ ~ Uniform(5, 50)
[ks_results_lambda_low, ad_results_lambda_low] =...
    perform_ks_ad_tests_lambda(metrics_low.lambda, display_results);
[ks_results_lambda_high, ad_results_lambda_high] =...
    perform_ks_ad_tests_lambda(metrics_high.lambda, display_results);

% τ_0 ~ Uniform(50, 150)
[ks_results_tau_0_low, ad_results_tau_0_low] =...
    perform_ks_ad_tests_tau_0(metrics_low.tau_0, display_results);
[ks_results_tau_0_high, ad_results_tau_0_high] =...
    perform_ks_ad_tests_tau_0(metrics_high.tau_0, display_results);

% low SNR ~ Unifrom(0, 10)
[ks_results_snr_low, ad_results_snr_low] =...
    perform_ks_ad_tests_snr(metrics_low.snr, 'low', display_results);

% high SNR ~ Unifrom(10, 30)
[ks_results_snr_high, ad_results_snr_high] =...
    perform_ks_ad_tests_snr(metrics_high.snr, 'high', display_results);

plot_ks_ad_results(ks_results_lambda_low, ks_results_lambda_high, ...
                  ad_results_lambda_low, ad_results_lambda_high, ...
                  ks_results_tau_0_low, ks_results_tau_0_high, ...
                  ad_results_tau_0_low, ad_results_tau_0_high, ...
                  ks_results_snr_low, ks_results_snr_high, ...
                  ad_results_snr_low, ad_results_snr_high, ...
                  'λ_τ0_SNR_tests_verification', display_plots);

% τ_l = τ_0+X_l, X_l ~ Poisson Process(L/λ)
tau_l_test_results =...
    perform_tau_l_verification(num_iterations, num_samples, display_results);

plot_tau_l_test_results(tau_l_test_results,'tau_l_results',display_plots);

%%%%%%%%%%% Chi-squared GOF test %%%%%%%%%%%
 
% L ~ Discrete_Uniform(3, 15)
L_GOF_results = perform_L_verification(num_iterations, num_samples, display_results);

plot_L_GOF_test_results(L_GOF_results, 'L_GOF_test', display_plots);

%%%%%%%%%%% Mean, t-test & variance tests %%%%%%%%%%%

% Tap Power, tap variance & Scaled Sum of taps
[tap_power_test_results] =...
    perform_tap_power_verification(num_iterations, num_samples, display_results);

plot_tap_power_test_results(tap_power_test_results, 'tap_power_tests_verification', display_plots)

% Noise variance 
[noise_variance_test_results] =...
     perform_noise_verification(num_iterations, num_samples, display_results);

plot_noise_test_results(noise_variance_test_results, 'noise_variance_test', display_plots);

%%%%%%%%%%% Helper-functions %%%%%%%%%%%

function metrics = calculate_metrics(L, lambda, tau_0, tau_l, E_hl_2, sigma_l_2, h, w, sigma_w_2, snr)
% Calculate and organize metrics for a given SNR case
    % Inputs:
    %   L           - Number of taps
    %   lambda      - Arrival rate
    %   tau_0       - Arrival time of the first tap (ToA)
    %   tau_l       - Time delay of the l-th tap
    %   E_hl_2      - Expected power of each tap
    %   sigma_l_2   - Tap variance
    %   h           - Channel coefficients
    %   w           - Noise vector
    %   sigma_w_2   - Noise variance
    %   snr         - Signal-to-noise ratio used
    % Outputs:
    %   metrics     - Structure containing all organized and calculated metrics

     % Extract non-NaN values and flatten into a single vector
    tau_l_flat = tau_l(~isnan(tau_l));          
    E_hl_2_flat = E_hl_2(~isnan(E_hl_2));     
    sigma_l_2_flat = sigma_l_2(~isnan(sigma_l_2)); 

    % Organize and calculate metrics
    metrics.L = L;
    metrics.lambda = lambda;
    metrics.tau_0 = tau_0;
    metrics.tau_l = tau_l_flat;
    metrics.E_hl_2 = E_hl_2_flat;
    metrics.sigma_l_2 = sigma_l_2_flat;
    metrics.snr = snr;
    
    % Sum of scaled l-th tap channel coefficients (∑(|h̃_l|^2))
    h(isnan(h)) = 0; % Treat NaNs as zeros
    metrics.sum_scaled_tap_coeff = sum(abs(h).^2);

    % Difference between the AWGN frequency vector variance and noise variance (Var(w⃗) - σ_w^2)
    Var_w = var(w); % Empirical variance of the noise vector
    metrics.Var_diff_w = Var_w - sigma_w_2;
end

function plot_cdf_pdf(metrics_high, metrics_low, save_folder, display_plots)
% Generate CDF and PDF plots for high and low snr metrics
    % Inputs:
    %   metrics_high  - metrics of the high SNR case
    %   metrics_low   - metrics of the high SNR case
    %   save_folder   - Folder name to save the plots in it
    %   display_plots - true to display plots or false to not display
    % Outputs:
    %   Folder with 9 plots of CDF & PDF for high & low SNR cases

    % List of metrics to be plotted
    metric_names = {'L', 'lambda', 'tau_0', 'tau_l', 'E_hl_2', 'sigma_l_2', ...
                    'sum_scaled_tap_coeff', 'snr', 'Var_diff_w'};
    % Titles for the figures
    titles = {'Number of Taps (L)', 'Arrival Rate (λ)', 'First Tap Arrival Time (τ_0)', ...
              'Tap Delays (τ_l)', 'Expected Power of Taps (E[|h_l|^2])', 'Tap Variance (σ_l^2)', ...
              'Sum of Scaled Tap Coefficients (∑(|h̃_l|^2))', 'SNR', 'Variance of Noise Difference (Var(w) - σ_w^2)'};
    
    % Ensure save folder exists
    if ~exist(save_folder, 'dir')
        mkdir(save_folder);
    end

    % Create figures for each metric
    for i = 1:length(metric_names)
        % Extract high and low SNR data
        metric_high = metrics_high.(metric_names{i});
        metric_low = metrics_low.(metric_names{i});
        
        % Create a new figure
        if display_plots
            figure;
        else
            figure('Visible', 'off');
        end
        sgtitle(titles{i});

        % PDF for high SNR case
        subplot(2, 2, 1);
        histogram(metric_high, 'Normalization', 'pdf');
        title('PDF - High SNR');
        xlabel('Value');
        ylabel('Probability Density');

        % CDF for high SNR case
        subplot(2, 2, 2);
        histogram(metric_high, 'Normalization', 'cdf');
        title('CDF - High SNR');
        xlabel('Value');
        ylabel('Cumulative Probability');

        % PDF for low SNR case
        subplot(2, 2, 3);
        histogram(metric_low, 'Normalization', 'pdf');
        title('PDF - Low SNR');
        xlabel('Value');
        ylabel('Probability Density');

        % CDF for low SNR case
        subplot(2, 2, 4);
        histogram(metric_low, 'Normalization', 'cdf');
        title('CDF - Low SNR');
        xlabel('Value');
        ylabel('Cumulative Probability');

        % Save the figure
        saveas(gcf, fullfile(save_folder, [metric_names{i} '.tiff']));

        % Close the figure if not displaying
        if ~display_plots
            close(gcf);
        end
    end
end

function [ks_results_lambda, ad_results_lambda] = perform_ks_ad_tests_lambda(empirical_lambda_val, display_results)
    % Perform KS and AD tests on the empirical lambda data
    % Inputs:
    %   empirical_lambda_val - Array of empirical lambda values
    %   display_results      - Boolean to decide if results should be printed (true/false)
    % Outputs:
    %   ks_results_lambda - Structure containing KS test results
    %   ad_results_lambda - Structure containing AD test results

    % Define the theoretical distribution for lambda ~ Uniform(5, 50)
    lambda_dist = makedist('Uniform', 'lower', 5, 'upper', 50);

    % Kolmogorov-Smirnov Test
    [h_ks, p_ks, ks_stat] = kstest(empirical_lambda_val, 'CDF', lambda_dist);
    
    % Anderson-Darling Test
    [h_ad, p_ad, ad_stat] = adtest(empirical_lambda_val, 'Distribution', lambda_dist);

    % Store results in structures
    ks_results_lambda = struct('hypothesis_rejected', h_ks, 'p_value', p_ks, 'ks_statistic', ks_stat);
    ad_results_lambda = struct('hypothesis_rejected', h_ad, 'p_value', p_ad, 'ad_statistic', ad_stat);

    % Optionally display results
    if display_results
        fprintf('KS Test Results for Lambda:\n');
        fprintf('Hypothesis rejected (h): %d\n', h_ks);
        fprintf('p-value: %f\n', p_ks);
        fprintf('KS Statistic: %f\n', ks_stat);

        fprintf('\nAD Test Results for Lambda:\n');
        fprintf('Hypothesis rejected (h): %d\n', h_ad);
        fprintf('p-value: %f\n', p_ad);
        fprintf('AD Statistic: %f\n', ad_stat);
    end
end

function [ks_results_tau_0, ad_results_tau_0] = perform_ks_ad_tests_tau_0(empirical_tau_0_val, display_results)
    % Perform KS and AD tests on the empirical tau_0 data
    % Inputs:
    %   empirical_tau_0_val - Array of empirical tau_0 values
    %   display_results      - Boolean to decide if results should be printed (true/false)
    % Outputs:
    %   ks_results_tau_0 - Structure containing KS test results
    %   ad_results_tau_0 - Structure containing AD test results

    % Define the theoretical distribution for tau_0 ~ Uniform(50, 150)
    tau_0_dist = makedist('Uniform', 'lower', 50, 'upper', 150);

    % Kolmogorov-Smirnov Test
    [h_ks, p_ks, ks_stat] = kstest(empirical_tau_0_val, 'CDF', tau_0_dist);
    
    % Anderson-Darling Test
    [h_ad, p_ad, ad_stat] = adtest(empirical_tau_0_val, 'Distribution', tau_0_dist);

    % Store results in structures
    ks_results_tau_0 = struct('hypothesis_rejected', h_ks, 'p_value', p_ks, 'ks_statistic', ks_stat);
    ad_results_tau_0 = struct('hypothesis_rejected', h_ad, 'p_value', p_ad, 'ad_statistic', ad_stat);

    % Optionally display results
    if display_results
        fprintf('KS Test Results for tau_0:\n');
        fprintf('Hypothesis rejected (h): %d\n', h_ks);
        fprintf('p-value: %f\n', p_ks);
        fprintf('KS Statistic: %f\n', ks_stat);

        fprintf('\nAD Test Results for tau_0:\n');
        fprintf('Hypothesis rejected (h): %d\n', h_ad);
        fprintf('p-value: %f\n', p_ad);
        fprintf('AD Statistic: %f\n', ad_stat);
    end
end

function [ks_results_snr, ad_results_snr] = perform_ks_ad_tests_snr(empirical_snr_val, snr_case, display_results)
    % Perform KS and AD tests on the empirical SNR data
    % Inputs:
    %   empirical_snr_val - Array of empirical SNR values
    %   snr_case          - 'high' or 'low' to specify the SNR case
    %   display_results   - Boolean to decide if results should be printed (true/false)
    % Outputs:
    %   ks_results_snr - Structure containing KS test results
    %   ad_results_snr - Structure containing AD test results

    % Define the theoretical distribution for SNR
    if strcmp(snr_case, 'high')
        snr_dist = makedist('Uniform', 'lower', 10, 'upper', 30);
    elseif strcmp(snr_case, 'low')
        snr_dist = makedist('Uniform', 'lower', 0, 'upper', 10);
    else
        error('Invalid SNR case. Use ''high'' or ''low''.');
    end

    % Kolmogorov-Smirnov Test
    [h_ks, p_ks, ks_stat] = kstest(empirical_snr_val, 'CDF', snr_dist);
    
    % Anderson-Darling Test
    [h_ad, p_ad, ad_stat] = adtest(empirical_snr_val, 'Distribution', snr_dist);

    % Store results in structures
    ks_results_snr = struct('hypothesis_rejected', h_ks, 'p_value', p_ks, 'ks_statistic', ks_stat);
    ad_results_snr = struct('hypothesis_rejected', h_ad, 'p_value', p_ad, 'ad_statistic', ad_stat);

    % Optionally display results
    if display_results
        fprintf('KS Test Results for SNR (%s case):\n', snr_case);
        fprintf('Hypothesis rejected (h): %d\n', h_ks);
        fprintf('p-value: %f\n', p_ks);
        fprintf('KS Statistic: %f\n', ks_stat);

        fprintf('\nAD Test Results for SNR (%s case):\n', snr_case);
        fprintf('Hypothesis rejected (h): %d\n', h_ad);
        fprintf('p-value: %f\n', p_ad);
        fprintf('AD Statistic: %f\n', ad_stat);
    end
end

function plot_ks_ad_results(ks_results_lambda_low, ks_results_lambda_high, ...
                            ad_results_lambda_low, ad_results_lambda_high, ...
                            ks_results_tau_0_low, ks_results_tau_0_high, ...
                            ad_results_tau_0_low, ad_results_tau_0_high, ...
                            ks_results_snr_low, ks_results_snr_high, ...
                            ad_results_snr_low, ad_results_snr_high, ...
                            folder_name, display_plots)

    % Ensure save folder exists
    if ~exist(folder_name, 'dir')
        mkdir(folder_name);
    end

    % Create figures for KS and AD test results
    
    % Define thresholds
    ks_thresholds = [1.354, 1.617]; % 95% and 99% thresholds for KS test
    ad_thresholds = [3.788, 4.882]; % 95% and 99% thresholds for AD test
    thresholds_labels = {'95%', '99%'};

    % Prepare data for plotting
    parameters = {'\lambda', '\tau_0', 'SNR'};
    
    % Prepare KS Test Results
    ks_statistics_low = [ks_results_lambda_low.ks_statistic; ...
                         ks_results_tau_0_low.ks_statistic; ...
                         ks_results_snr_low.ks_statistic];
    ks_statistics_high = [ks_results_lambda_high.ks_statistic; ...
                          ks_results_tau_0_high.ks_statistic; ...
                          ks_results_snr_high.ks_statistic];

    % Prepare AD Test Results
    ad_statistics_low = [ad_results_lambda_low.ad_statistic; ...
                         ad_results_tau_0_low.ad_statistic; ...
                         ad_results_snr_low.ad_statistic];
    ad_statistics_high = [ad_results_lambda_high.ad_statistic; ...
                          ad_results_tau_0_high.ad_statistic; ...
                          ad_results_snr_high.ad_statistic];

    % Plotting KS Test Results
    figure;
    
    % Low SNR case
    subplot(1, 2, 1);
    hold on;
    for i = 1:length(parameters)
        t_val = ks_statistics_low(i);
        if t_val < ks_thresholds(1)
            plot_color = [0.4660, 0.6740, 0.1880]; % Green
        elseif t_val < ks_thresholds(2)
            plot_color = [0.9290, 0.6940, 0.1250]; % Yellow
        else
            plot_color = [0.6350, 0.0780, 0.1840]; % Red
        end
        plot(i, t_val, 'o', 'MarkerEdgeColor', plot_color, 'MarkerFaceColor', plot_color);
    end
    % Add threshold lines
    yline(ks_thresholds(1), '--r', thresholds_labels{1});
    yline(ks_thresholds(2), '--b', thresholds_labels{2});
    title('KS Test Results (Low SNR)');
    xticks(1:length(parameters));
    xticklabels(parameters);
    xlabel('Parameter');
    ylabel('KS Statistic');
    grid on;
    hold off;

    % High SNR case
    subplot(1, 2, 2);
    hold on;
    for i = 1:length(parameters)
        t_val = ks_statistics_high(i);
        if t_val < ks_thresholds(1)
            plot_color = [0.4660, 0.6740, 0.1880]; % Green
        elseif t_val < ks_thresholds(2)
            plot_color = [0.9290, 0.6940, 0.1250]; % Yellow
        else
            plot_color = [0.6350, 0.0780, 0.1840]; % Red
        end
        plot(i, t_val, 'o', 'MarkerEdgeColor', plot_color, 'MarkerFaceColor', plot_color);
    end
    % Add threshold lines
    yline(ks_thresholds(1), '--r', thresholds_labels{1});
    yline(ks_thresholds(2), '--b', thresholds_labels{2});
    title('KS Test Results (High SNR)');
    xticks(1:length(parameters));
    xticklabels(parameters);
    xlabel('Parameter');
    ylabel('KS Statistic');
    grid on;
    hold off;
    
    % Save the KS figure
    saveas(gcf, fullfile(folder_name, 'ks_test_results.tiff'));

    % Plotting AD Test Results
    figure;
    
    % Low SNR case
    subplot(1, 2, 1);
    hold on;
    for i = 1:length(parameters)
        t_val = ad_statistics_low(i);
        if t_val < ad_thresholds(1)
            plot_color = [0.4660, 0.6740, 0.1880]; % Green
        elseif t_val < ad_thresholds(2)
            plot_color = [0.9290, 0.6940, 0.1250]; % Yellow
        else
            plot_color = [0.6350, 0.0780, 0.1840]; % Red
        end
        plot(i, t_val, 'o', 'MarkerEdgeColor', plot_color, 'MarkerFaceColor', plot_color);
    end
    % Add threshold lines
    yline(ad_thresholds(1), '--r', thresholds_labels{1});
    yline(ad_thresholds(2), '--b', thresholds_labels{2});
    title('AD Test Results (Low SNR)');
    xticks(1:length(parameters));
    xticklabels(parameters);
    xlabel('Parameter');
    ylabel('AD Statistic');
    grid on;
    hold off;

    % High SNR case
    subplot(1, 2, 2);
    hold on;
    for i = 1:length(parameters)
        t_val = ad_statistics_high(i);
        if t_val < ad_thresholds(1)
            plot_color = [0.4660, 0.6740, 0.1880]; % Green
        elseif t_val < ad_thresholds(2)
            plot_color = [0.9290, 0.6940, 0.1250]; % Yellow
        else
            plot_color = [0.6350, 0.0780, 0.1840]; % Red
        end
        plot(i, t_val, 'o', 'MarkerEdgeColor', plot_color, 'MarkerFaceColor', plot_color);
    end
    % Add threshold lines
    yline(ad_thresholds(1), '--r', thresholds_labels{1});
    yline(ad_thresholds(2), '--b', thresholds_labels{2});
    title('AD Test Results (High SNR)');
    xticks(1:length(parameters));
    xticklabels(parameters);
    xlabel('Parameter');
    ylabel('AD Statistic');
    grid on;
    hold off;
    
    % Save the AD figure
    saveas(gcf, fullfile(folder_name, 'ad_test_results.tiff'));

    % Close figures if not displaying
    if ~display_plots
        close all;
    end
end

function [test_results] = perform_tau_l_verification(num_iterations, num_samples, display_results)
    % Perform verification for tau_l 
    % Inputs:
    %   num_iterations - Number τ_l values to test (e.g., 100)
    %   num_samples    - Number of samples to generate for each value (e.g., 100,000)
    %   display_results - Boolean to decide if results should be printed (true/false)
    % Outputs:
    %   test_results   - Structure containing the results of KS & AD tests

    % Pre-allocate structure to store test results
    test_results(num_iterations).ks_statistic_poisson = [];
    test_results(num_iterations).ks_statistic_exponential = [];
    test_results(num_iterations).ad_statistic_exponential = [];
    test_results(num_iterations).mean_error_poisson = [];
    test_results(num_iterations).mean_error_exponential = [];

    % Initialize wait bar
    h = waitbar(0, 'Performing tau_l KS & AD tests');

    % Perform tests
    for i = 1:num_iterations

        % Update wait bar
        waitbar(i / num_iterations, h, sprintf('Performing tau_l KS & AD tests (Iteration %d of %d)', i, num_iterations));

        % Step 1: Generate lambda from L & time_interval
        L = randi([3 15]); % Number of taps
        time_interval = 5 + (45) * rand; % Arrival time interval
        lambda = L / time_interval;

        % Step 2: Generate Poisson Process with #num_samples samples
        U= 1.0 - rand(1,num_samples);
        next_time_all =cumsum(-log(U+eps)/lambda);
        next_time = next_time_all-next_time_all(1);

        for T=1:(num_samples-100)
            idx = find(next_time < time_interval);
            N_Events_in_ime_Interval(T) = length(idx)-1;
            if idx(end) == length(next_time)
                break
            else
                next_time=next_time_all((T+1):end)-next_time_all(T+1);
                if length(next_time)<1
                    break
                end
            end
        end
        
        % Step 3: From process define the Counting process (number of events) and the time between them (inter arrival times)
        num_of_arrival_poisson = N_Events_in_ime_Interval;
        inter_arrival_times_exponential = diff(next_time_all);

        % Calculate the means for them
        num_of_arrival_mean = mean(N_Events_in_ime_Interval);    % mean of the number of events in time interval
        inter_arrival_times_mean = mean(diff(next_time_all));    % mean of the exponential random variable

        % Step 4: Perform KS and AD tests to check if samples follow the poisson and exponential distribution
        poisson_dist = makedist('Poisson', 'lambda', L);

        % if i == 1
        %     figure
        %     hold on
        %     empirical_cdf = histogram(num_of_arrival_poisson, 'Normalization','cdf')
        %     theoretical_cdf = plot(poisson_dist,'PlotType','cdf')
        %     hold off
        %     legend('empirical','theory')
        %     grid on
        % 
        %     empirical_values = empirical_cdf.Values;
        %     empirical_x_values = empirical_cdf.BinEdges(1:end-1);
        % 
        %     x_values = min(num_of_arrival_poisson):max(num_of_arrival_poisson);
        %     theoretical_values = cdf(poisson_dist, x_values);
        % 
        %     difference_values = theoretical_values - empirical_values;
        % 
        %     figure
        %     hold on
        %     difference_plot = stem(x_values, difference_values);
        %     hold off
        %     legend('difference')
        %     grid on
        % end

        [~, ~, ks_result_poisson] = kstest(num_of_arrival_poisson, 'CDF', poisson_dist);  % KS test poisson
        test_results(i).ks_statistic_poisson = ks_result_poisson;

        exponential_dist = makedist('Exponential','mu', 1/lambda);

        [~, ~, ks_result_exponential] = kstest(inter_arrival_times_exponential, 'CDF', exponential_dist);  % KS test exponential
        [~, ~, ad_result_exponential] = adtest(inter_arrival_times_exponential, 'Distribution', exponential_dist);  % AD test exponential
        test_results(i).ks_statistic_exponential = ks_result_exponential;
        test_results(i).ad_statistic_exponential = ad_result_exponential;

        % Step 5: Calculate mean error
        mean_error_poisson = abs(L - num_of_arrival_mean);
        mean_error_exponential = abs((1/lambda)-inter_arrival_times_mean);

        test_results(i).mean_error_poisson = mean_error_poisson;
        test_results(i).mean_error_exponential = mean_error_exponential;

        % Display results if required
        if display_results
            fprintf('KS Test Statistic Poisson: %s\n', num2str(ks_result_poisson));
            fprintf('KS Test Statistic Exponential: %s\n', num2str(ks_result_exponential));
            fprintf('AD Test Statistic Exponential: %s\n', num2str(ad_result_exponential));
            fprintf('Mean Error of Poisson arrival rate: %s\n', num2str(mean_error_poisson));
            fprintf('Mean Error of Exponential mean: %s\n', num2str(mean_error_exponential));
            fprintf('--------------------------\n');
        end
    end

    % Close the wait bar
    close(h);
end

function plot_tau_l_test_results(test_results, folder_name, display_plots)
    % Plot the KS, AD & mean test results 
    % Inputs:
    %   test_results  - Structure containing the results of the test
    %   folder_name   - Folder to save the plots
    %   display_plots - Boolean to display the plot (true/false)

    % Ensure save folder exists
    if ~exist(folder_name, 'dir')
        mkdir(folder_name);
    end

    % Get the number of iterations
    num_iterations = length(test_results);

    % Define thresholds
    ks_thresholds = [1.354, 1.617]; % 95% and 99% thresholds for KS test
    ad_thresholds = [3.788, 4.882]; % 95% and 99% thresholds for AD test

    % ----------- 1. Plot the KS test results for tau_l (Poisson) ----------- %
    figure;
    hold on;
    for iter = 1:num_iterations
        ks_val = test_results(iter).ks_statistic_poisson;
        if ks_val < ks_thresholds(1)
            plot_color = [0.4660 0.6740 0.1880]; % Green for below 95% threshold
        elseif ks_val < ks_thresholds(2)
            plot_color = [0.9290 0.6940 0.1250]; % Yellow for between 95% and 99% thresholds
        else
            plot_color = [0.6350 0.0780 0.1840]; % Red for above 99% threshold
        end
        plot(iter, ks_val, 'o', 'MarkerEdgeColor', plot_color, 'MarkerFaceColor', plot_color);
    end
    yline(ks_thresholds(1), '--r', '95% Threshold');
    yline(ks_thresholds(2), '--b', '99% Threshold');
    title('KS statistic for \tau_l Poisson');
    xlabel('Iteration');
    ylabel('KS-Statistic');
    grid on;
    hold off;
    saveas(gcf, fullfile(folder_name, 'ks_test_poisson.tiff')); % Save as .tiff
    if ~display_plots
        close(gcf); % Close the plot if not displaying
    end

    % ----------- 2. Plot the KS test results for tau_l (Exponential) ----------- %
    figure;
    hold on;
    for iter = 1:num_iterations
        ks_val = test_results(iter).ks_statistic_exponential;
        if ks_val < ks_thresholds(1)
            plot_color = [0.4660 0.6740 0.1880]; % Green for below 95% threshold
        elseif ks_val < ks_thresholds(2)
            plot_color = [0.9290 0.6940 0.1250]; % Yellow for between 95% and 99% thresholds
        else
            plot_color = [0.6350 0.0780 0.1840]; % Red for above 99% threshold
        end
        plot(iter, ks_val, 'o', 'MarkerEdgeColor', plot_color, 'MarkerFaceColor', plot_color);
    end
    yline(ks_thresholds(1), '--r', '95% Threshold');
    yline(ks_thresholds(2), '--b', '99% Threshold');
    title('KS statistic for \tau_l Exponential');
    xlabel('Iteration');
    ylabel('KS-Statistic');
    grid on;
    hold off;
    saveas(gcf, fullfile(folder_name, 'ks_test_exponential.tiff')); % Save as .tiff
    if ~display_plots
        close(gcf); % Close the plot if not displaying
    end

    % ----------- 3. Plot the AD test results for tau_l (Exponential) ----------- %
    figure;
    hold on;
    for iter = 1:num_iterations
        ad_val = test_results(iter).ad_statistic_exponential;
        if ad_val < ad_thresholds(1)
            plot_color = [0.4660 0.6740 0.1880]; % Green for below 95% threshold
        elseif ad_val < ad_thresholds(2)
            plot_color = [0.9290 0.6940 0.1250]; % Yellow for between 95% and 99% thresholds
        else
            plot_color = [0.6350 0.0780 0.1840]; % Red for above 99% threshold
        end
        plot(iter, ad_val, 'o', 'MarkerEdgeColor', plot_color, 'MarkerFaceColor', plot_color);
    end
    yline(ad_thresholds(1), '--r', '95% Threshold');
    yline(ad_thresholds(2), '--b', '99% Threshold');
    title('AD statistic for \tau_l Exponential');
    xlabel('Iteration');
    ylabel('AD-Statistic');
    grid on;
    hold off;
    saveas(gcf, fullfile(folder_name, 'ad_test_exponential.tiff')); % Save as .tiff
    if ~display_plots
        close(gcf); % Close the plot if not displaying
    end

    % ----------- 4. Plot the Error Mean Arrival Rate ----------- %
    figure;
    hold on;
    for iter = 1:num_iterations
        plot(iter, test_results(iter).mean_error_poisson, 'o-', 'LineWidth', 1.5); % Mean Error per iteration
    end
    title('Error of Mean Arrival Rate');
    xlabel('Iteration');
    ylabel('Error');
    grid on;
    hold off;
    saveas(gcf, fullfile(folder_name, 'mean_arrival_rate_error.tiff')); % Save as .tiff
    if ~display_plots
        close(gcf); % Close the plot if not displaying
    end

    % ----------- 5. Plot the Error Mean Exponential ----------- %
    figure;
    hold on;
    for iter = 1:num_iterations
        plot(iter, test_results(iter).mean_error_exponential, 'o-', 'LineWidth', 1.5); % Mean Error per iteration
    end
    title('Error of Mean for Exponential Dist.');
    xlabel('Iteration');
    ylabel('Error');
    grid on;
    hold off;
    saveas(gcf, fullfile(folder_name, 'mean_exponential_error.tiff')); % Save as .tiff
    if ~display_plots
        close(gcf); % Close the plot if not displaying
    end
end

function [test_results] = perform_tap_power_verification(num_iterations, num_samples, display_results)
    % Perform verification for [|h_l|^2], σ_l^2, and sum of scaled tap power
    % Inputs:
    %   num_iterations - Number of L and τ_l combinations to test (e.g., 100)
    %   num_samples    - Number of samples to generate for each combination (e.g., 100,000)
    %   display_results - Boolean to decide if results should be printed (true/false)
    % Outputs:
    %   test_results   - Structure containing the results of the mean, variance, t-tests, and f-tests
    
    tau_rms = 75; % RMS delay spread (in nsec)

    % Pre-allocate structure to store test results
    test_results(num_iterations).mean_test_hl = [];
    test_results(num_iterations).t_test_hl = [];
    test_results(num_iterations).f_test_hl = [];
    test_results(num_iterations).mean_test_scaled_sum = [];
    test_results(num_iterations).t_test_scaled_sum = [];

    % Step 1: Select L and lambda, generate τ_l
    L = randi([3, 15]); % Number of taps
    time_interval = 5 + (45) * rand; % Arrival rate interval
    lambda = L / time_interval;
    tau_l = zeros(1,L);
    for k = 2:L
        U = 1.0 -rand(1);
        next_time = -log(U)/lambda;
        tau_l(k) = tau_l(k-1) + next_time;
    end

    % Step 2: Calculate expected power and variance
    E_hl_2 = exp(-tau_l / tau_rms); % Expected power E[|h_l|^2]
    sigma_l_2 = E_hl_2 / 2; % Tap variance

    % Initialize wait bar
    h = waitbar(0, 'Performing tap power verification tests');

    % Loop through each iteration to generate and verify
    for iter = 1:num_iterations
        % Update wait bar
        waitbar(iter / num_iterations, h, sprintf('Performing tap power tests (Iteration %d of %d)', iter, num_iterations));

        % Step 3a: Generate 100,000 samples of h_l
        h_l_samples = sqrt(sigma_l_2) .* (randn(num_samples, L) + 1i * randn(num_samples, L)); % Complex Gaussian h_l

        % Step 3b: Calculate sample mean and variance of the generated h_l
        mean_hl_samples = mean(abs(h_l_samples).^2, 1); % Mean power of each tap
        var_hl_samples = var(h_l_samples, 0, 1) /2; % Variance of each tap

        % Step 3c: Calculate mean deviation test 
        mean_test_hl = mean_hl_samples - E_hl_2; % e(τ_l) = mean(E[|h_l|^2]) - exp(-τ_l/τ_rms)
        mean_test_hl = abs(mean_test_hl);

        % Step 3d: Perform t-test for mean
        t_test_hl = zeros(1, L); % Pre-allocate array for t-statistics

        for l = 1:L
            % Perform t-test for each column in h_l_samples
            [~,~,~,t_test_hl_stats] = ttest(abs(h_l_samples(:, l)).^2, E_hl_2(l)); % Perform t-test for tap l
           t_test_hl(l) = t_test_hl_stats.tstat; % Store t-statistic for tap l
        end

        t_test_hl = abs(t_test_hl);

        % Step 3e: Perform f-test for variance
        f_test_hl = var_hl_samples ./ sigma_l_2; % f-test compares variance to σ_l^2
        f_test_hl(f_test_hl < 1) = 1 ./ f_test_hl(f_test_hl < 1);

        % Step 3f: Calculate variance of the sample mean using two methods
        n_values = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]; % Number of samples for each iteration
        expected_value = E_hl_2; % Theoretical expected value of |h_l|^2 for each tap

        % Pre-allocate storage for both variance methods
        variance_deviation = zeros(length(n_values), L);  % Using deviation from expected value
        variance_estimation = zeros(length(n_values), L); % Using variance of |h_l|^2 / n

        for l = 1:L
            h_l_squared = abs(h_l_samples(:, l)).^2; % |h_l|^2 for all 100,000 samples of tap l

            for idx = 1:length(n_values)
                n = n_values(idx);  % Select the first 'n' samples

                % Use the first 'n' samples of |h_l|^2
                h_l_squared_n = h_l_squared(1:n);

                % Method 1: Calculate the sample mean and deviation from the theoretical mean
                sample_mean = mean(h_l_squared_n);
                deviation = (sample_mean - expected_value(l))^2;  % Squared deviation from expected value

                % Store the variance using deviation from expected value
                variance_deviation(idx, l) = deviation;

                % Method 2: Calculate the variance of |h_l|^2 and divide by n
                variance_hl_squared = var(h_l_squared_n);  % Variance of |h_l|^2 for n samples
                variance_estimation(idx, l) = variance_hl_squared / n;  % Estimate of variance of the sample mean
            end
        end

        % Step 4a: Scale the 100,000 h_l taps to create scaled taps
        total_power = sum(E_hl_2); % Total power
        h_l_scaled = h_l_samples / sqrt(total_power); % Scale each coefficient

        % Step 4b: Calculate sum of scaled power across all samples
        scaled_power_sum = sum(abs(h_l_scaled).^2, 2); % Sum across all taps for each sample (100,000 sums)

        % Step 4c: Calculate mean and variance of the sum of scaled tap power
        mean_scaled_sum = mean(scaled_power_sum); % Mean of the scaled sum

        % Step 5: Calculate mean error for the sum of scaled tap power
        mean_test_scaled_sum = mean_scaled_sum - 1; % Expected sum is 1 (normalized)
        mean_test_scaled_sum = abs(mean_test_scaled_sum);

        % Step 6: T-test for sum
        [~,~,~,t_test_scaled_sum_stats] = ttest(scaled_power_sum, 1); % Test if mean is 1
        t_test_scaled_sum = t_test_scaled_sum_stats.tstat; % Extract t-statistic
        t_test_scaled_sum = abs(t_test_scaled_sum);

        % Step 7: Calculate variance of sum of scaled taps as a function of sample size
        n_values = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]; % Sample sizes
        expected_value_scaled_sum = 1; % Theoretical expected value of the scaled sum

        % Pre-allocate storage for variance calculations for scaled sum
        variance_deviation_scaled = zeros(length(n_values), 1);  % Using deviation from expected value
        variance_estimation_scaled = zeros(length(n_values), 1); % Using variance of sum / n

        for idx = 1:length(n_values)
            n = n_values(idx);  % Select the first 'n' samples

            % Use the first 'n' samples of the scaled sum of power
            scaled_power_sum_n = scaled_power_sum(1:n);

            % Method 1: Calculate the sample mean and deviation from the theoretical mean
            sample_mean_scaled = mean(scaled_power_sum_n);
            deviation_scaled = (sample_mean_scaled - expected_value_scaled_sum)^2;  % Squared deviation from expected value

            % Store the variance using deviation from expected value
            variance_deviation_scaled(idx) = deviation_scaled;

            % Method 2: Calculate the variance of the sum of scaled taps and divide by n
            variance_scaled_sum_n = var(scaled_power_sum_n);  % Variance of the sum for n samples
            variance_estimation_scaled(idx) = variance_scaled_sum_n / n;  % Estimate of variance of the sample mean
        end

        % Store variance results in the test_results structure
        test_results(iter).variance_deviation_scaled = variance_deviation_scaled;
        test_results(iter).variance_estimation_scaled = variance_estimation_scaled;

        % Step 8: Store results in structure
        test_results(iter).mean_test_hl = mean_test_hl;
        test_results(iter).t_test_hl = t_test_hl;
        test_results(iter).f_test_hl = f_test_hl;
        test_results(iter).variance_deviation = variance_deviation;
        test_results(iter).variance_estimation = variance_estimation;
        test_results(iter).mean_test_scaled_sum = mean_test_scaled_sum;
        test_results(iter).t_test_scaled_sum = t_test_scaled_sum;
        test_results(iter).variance_deviation_scaled = variance_deviation_scaled;
        test_results(iter).variance_estimation_scaled = variance_estimation_scaled;

        % Optionally display results
        if display_results
            fprintf('Mean Test E[|h_l|^2]: %s\n', num2str(mean_test_hl'));
            fprintf('T-Test E[|h_l|^2]: %s\n', num2str(t_test_hl'));
            fprintf('F-Test E[|h_l|^2]: %s\n', num2str(f_test_hl'));
            fprintf('Mean Test Sum of Scaled Power: %f\n', mean_test_scaled_sum);
            fprintf('Variance Test Sum of Scaled Power: %f\n', variance_estimation_scaled);
            fprintf('T-Test Sum of Scaled Power: %f\n', t_test_scaled_sum);
            fprintf('--------------------------\n');
        end
    end

    % Close the wait bar
    close(h);
end

function plot_tap_power_test_results(test_results, folder_name, display_plots)
    % Plot the mean error, t-statistic, and f-statistic across all iterations
    % Inputs:
    %   test_results  - Structure containing the results of the test
    %   folder_name   - Folder to save the plots
    %   display_plots - Boolean to display the plot (true/false)

    % Ensure save folder exists
    if ~exist(folder_name, 'dir')
        mkdir(folder_name);
    end

    % Get the number of iterations
    num_iterations = length(test_results);

    % ----------- 1. Plot the Mean Error ----------- %
    figure;
    hold on;
    for iter = 1:num_iterations
        plot(iter, test_results(iter).mean_test_hl, 'o-', 'LineWidth', 1.5); % Mean Error per iteration
    end
    title('Mean Error E[|h_l|^2] Across All Iterations');
    xlabel('Iteration');
    ylabel('Mean Error');
    grid on;
    hold off;
    saveas(gcf, fullfile(folder_name, 'mean_test_hl.tiff')); % Save as .tiff
    if ~display_plots
        close(gcf); % Close the plot if not displaying
    end

    % ----------- 2. Plot the T-Statistic for E[|h_l|^2] ----------- %
    t_threshold_95 = 1.96;
    t_threshold_99 = 2.5759;
    
    figure;
    hold on;
    for iter = 1:num_iterations
        for tap = 1:length(test_results(iter).t_test_hl)
            t_val = test_results(iter).t_test_hl(tap);
            if t_val < t_threshold_95
                plot_color = [0.4660 0.6740 0.1880]; % Green for below 95% threshold
            elseif t_val < t_threshold_99
                plot_color = [0.9290 0.6940 0.1250]; % Yellow for between 95% and 99% thresholds
            else
                plot_color = [0.6350 0.0780 0.1840]; % Red for above 99% threshold
            end
            plot(iter, t_val, 'o', 'MarkerEdgeColor', plot_color, 'MarkerFaceColor', plot_color);
        end
    end
    yline(t_threshold_95, '--r', '95% Threshold');
    yline(t_threshold_99, '--b', '99% Threshold');
    title('T-Statistic for E[|h_l|^2] Across All Iterations');
    xlabel('Iteration');
    ylabel('T-Statistic');
    grid on;
    hold off;
    saveas(gcf, fullfile(folder_name, 't_test_hl.tiff')); % Save as .tiff
    if ~display_plots
        close(gcf); % Close the plot if not displaying
    end

    % ----------- 3. Plot the F-Statistic for E[|h_l|^2] ----------- %
    f_threshold_95 = 1.0105;
    f_threshold_99 = 1.0148;

    figure;
    hold on;
    for iter = 1:num_iterations
        for tap = 1:length(test_results(iter).f_test_hl)
            f_val = test_results(iter).f_test_hl(tap);
            if f_val < f_threshold_95
                plot_color = [0.4660 0.6740 0.1880]; % Green for below 95% threshold
            elseif f_val < f_threshold_99
                plot_color = [0.9290 0.6940 0.1250]; % Yellow for between 95% and 99% thresholds
            else
                plot_color = [0.6350 0.0780 0.1840]; % Red for above 99% threshold
            end
            plot(iter, f_val, 'o', 'MarkerEdgeColor', plot_color, 'MarkerFaceColor', plot_color);
        end
    end
    yline(f_threshold_95, '--r', '95% Threshold');
    yline(f_threshold_99, '--b', '99% Threshold');
    title('F-Statistic for \sigma_l^2 Across All Iterations');
    xlabel('Iteration');
    ylabel('F-Statistic');
    grid on;
    hold off;
    saveas(gcf, fullfile(folder_name, 'f_test_hl.tiff')); % Save as .tiff
    if ~display_plots
        close(gcf); % Close the plot if not displaying
    end

    % ----------- 4. Plot the Variance Decay for E[|h_l|^2] ----------- %
    % Plot variance decay for E[|h_l|^2] for both deviation and estimation methods
    n_values = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000];  % X-axis for the plots

    figure;

    % Subplot 1: Variance Deviation Method
    subplot(1, 2, 1);  % Create a 1x2 grid and use the first subplot
    hold on;

    % Iterate over all test iterations and plot deviation method
    for iter = 1:num_iterations
        plot(n_values, test_results(iter).variance_deviation(:, 1), '-o', 'LineWidth', 1.5, ...
            'DisplayName', ['Iteration ', num2str(iter)]);
    end

    % Add labels, title, and settings for the deviation method plot
    xlabel('Number of Samples');
    ylabel('Variance of the Sample Mean');
    title('Variance Decay (Deviation Method)');
    grid on;
    set(gca, 'XScale', 'log');  % Use logarithmic scale for the x-axis
    legend('off');  % Disable the legend to avoid cluttering the plot

    hold off;

    % Subplot 2: Variance Estimation Method
    subplot(1, 2, 2);  % Use the second subplot in the 1x2 grid
    hold on;

    % Iterate over all test iterations and plot estimation method
    for iter = 1:num_iterations
        plot(n_values, test_results(iter).variance_estimation(:, 1), '-x', 'LineWidth', 1.5, ...
            'DisplayName', ['Iteration ', num2str(iter)]);
    end

    % Add labels, title, and settings for the estimation method plot
    xlabel('Number of Samples');
    ylabel('Variance of the Sample Mean');
    title('Variance Decay (Estimation Method)');
    grid on;
    set(gca, 'XScale', 'log');  % Use logarithmic scale for the x-axis
    legend('off');  % Disable the legend to avoid cluttering the plot

    hold off;

    % Add overall title to the figure
    sgtitle('Variance Decay Comparison: Deviation vs. Estimation Methods');

    % Save the figure as a .tiff file
    saveas(gcf, fullfile(folder_name, 'variance_decay_comparison.tiff'));  % Save as .tiff file

    % Close the figure if not displaying
    if ~display_plots
        close(gcf);
    end

    % ----------- 5. Plot the Mean Error (Scaled Sum) ----------- %
    figure;
    hold on;
    for iter = 1:num_iterations
        plot(iter, test_results(iter).mean_test_scaled_sum, 'o-', 'LineWidth', 1.5); % Mean Error per iteration
    end
    title('Mean Error Scaled Sum Across All Iterations');
    xlabel('Iteration');
    ylabel('Mean Error');
    grid on;
    hold off;
    saveas(gcf, fullfile(folder_name, 'mean_test_scaled_sum.tiff')); % Save as .tiff
    if ~display_plots
        close(gcf); % Close the plot if not displaying
    end

    % ----------- 6. Plot the T-Statistic for Scaled Sum ----------- %
    t_threshold_95 = 1.96;
    t_threshold_99 = 2.5759;
    
    figure;
    hold on;
    for iter = 1:num_iterations
        for tap = 1:length(test_results(iter).t_test_scaled_sum)
            t_val = test_results(iter).t_test_scaled_sum(tap);
            if t_val < t_threshold_95
                plot_color = [0.4660 0.6740 0.1880]; % Green for below 95% threshold
            elseif t_val < t_threshold_99
                plot_color = [0.9290 0.6940 0.1250]; % Yellow for between 95% and 99% thresholds
            else
                plot_color = [0.6350 0.0780 0.1840]; % Red for above 99% threshold
            end
            plot(iter, t_val, 'o', 'MarkerEdgeColor', plot_color, 'MarkerFaceColor', plot_color);
        end
    end
    yline(t_threshold_95, '--r', '95% Threshold');
    yline(t_threshold_99, '--b', '99% Threshold');
    title('T-Statistic for Scaled Sum Across All Iterations');
    xlabel('Iteration');
    ylabel('T-Statistic');
    grid on;
    hold off;
    saveas(gcf, fullfile(folder_name, 't_test_scaled_sum.tiff')); % Save as .tiff
    if ~display_plots
        close(gcf); % Close the plot if not displaying
    end

    % ----------- 7. Plot the Variance Decay for Scaled Sum ----------- %
    % Plot variance decay for Scaled sum for both deviation and estimation methods
    n_values = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000];  % X-axis for the plots

    figure;

    % Subplot 1: Variance Deviation Method
    subplot(1, 2, 1);  % Create a 1x2 grid and use the first subplot
    hold on;

    % Iterate over all test iterations and plot deviation method
    for iter = 1:num_iterations
        plot(n_values, test_results(iter).variance_deviation_scaled(:, 1), '-o', 'LineWidth', 1.5, ...
            'DisplayName', ['Iteration ', num2str(iter)]);
    end

    % Add labels, title, and settings for the deviation method plot
    xlabel('Number of Samples');
    ylabel('Variance of the mean of Scaled Sum');
    title('Variance Decay (Deviation Method)');
    grid on;
    set(gca, 'XScale', 'log');  % Use logarithmic scale for the x-axis
    legend('off');  % Disable the legend to avoid cluttering the plot

    hold off;

    % Subplot 2: Variance Estimation Method
    subplot(1, 2, 2);  % Use the second subplot in the 1x2 grid
    hold on;

    % Iterate over all test iterations and plot estimation method
    for iter = 1:num_iterations
        plot(n_values, test_results(iter).variance_estimation_scaled(:, 1), '-x', 'LineWidth', 1.5, ...
            'DisplayName', ['Iteration ', num2str(iter)]);
    end

    % Add labels, title, and settings for the estimation method plot
    xlabel('Number of Samples');
    ylabel('Variance of the mean of Scaled Sum');
    title('Variance Decay (Estimation Method)');
    grid on;
    set(gca, 'XScale', 'log');  % Use logarithmic scale for the x-axis
    legend('off');  % Disable the legend to avoid cluttering the plot

    hold off;

    % Add overall title to the figure
    sgtitle('Variance Decay Comparison: Deviation vs. Estimation Methods');

    % Save the figure as a .tiff file
    saveas(gcf, fullfile(folder_name, 'variance_decay_comparison_scaled.tiff'));  % Save as .tiff file

    % Close the figure if not displaying
    if ~display_plots
        close(gcf);
    end
end

function chi2_values = perform_L_verification(num_iterations, num_samples, display_results)
    % Perform Chi-squared GOF test for number of taps L
    % Inputs:
    %   num_iterations - Number of iterations for test (e.g., 100)
    %   num_samples    - Number of samples to generate for each iteration (e.g., 100,000)
    %   display_results - Boolean to decide if results should be printed (true/false)
    % Outputs:
    %   chi2_values   - results of the Chi-squared test values

    % Parameters
    L_min = 3;             % Minimum value of L
    L_max = 15;            % Maximum value of L
    num_categories = L_max - L_min + 1;  % Number of possible values for L (13)

    % Preallocate chi-square values array
    chi2_values = zeros(1, num_iterations);

    % Expected probabilities for uniform distribution
    expected_probs = ones(1, num_categories) / num_categories;
    
    % Bin edges for histogram (inclusive of edges at 3 and 15)
    bin_edges = L_min-0.5:L_max+0.5;
    
    % Loop over each iteration
    for iter = 1:num_iterations
        % Step 1: Generate random samples of L ~ UniformDiscrete(3,15)
        L_samples = randi([L_min, L_max], 1, num_samples);
        
        % Step 2: Perform chi-squared GOF test using chi2gof
        % chi2gof expects data in vector form, with expected probabilities
        [~, p, stats] = chi2gof(L_samples, 'Expected', expected_probs*num_samples, ...
            'Edges', bin_edges, 'Emin', 5);
        
        % Step 3: Extract and store the chi-square statistic
        chi2_values(iter) = stats.chi2stat;
    end
    
    % Display results if chosen
    if display_results
        disp('Chi-square values for each iteration:');
        disp(chi2_values);
    end
end

function plot_L_GOF_test_results(test_results, folder_name, display_plots)
    % Plot the Chi-squared GOF test results of number of taps L
    % Inputs:
    %   test_results  - Structure containing the results of the test
    %   folder_name   - Folder to save the plots
    %   display_plots - Boolean to display the plot (true/false)

    % Ensure save folder exists
    if ~exist(folder_name, 'dir')
        mkdir(folder_name);
    end

    % Get the number of iterations
    num_iterations = length(test_results);

    % ----------- 2. Plot the T-Statistic for E[|h_l|^2] ----------- %
    t_threshold_95 = 21.026;
    t_threshold_99 = 26.217;
    
    figure;
    hold on;
    for iter = 1:num_iterations
        t_val = test_results(iter);
        if t_val < t_threshold_95
            plot_color = [0.4660 0.6740 0.1880]; % Green for below 95% threshold
        elseif t_val < t_threshold_99
            plot_color = [0.9290 0.6940 0.1250]; % Yellow for between 95% and 99% thresholds
        else
            plot_color = [0.6350 0.0780 0.1840]; % Red for above 99% threshold
        end
        plot(iter, t_val, 'o', 'MarkerEdgeColor', plot_color, 'MarkerFaceColor', plot_color);
    end
    yline(t_threshold_95, '--r', '95% Threshold');
    yline(t_threshold_99, '--b', '99% Threshold');
    title('Chi-squared GOF Statistic for L Across All Iterations');
    xlabel('Iteration');
    ylabel('Chi-squared Statistic');
    grid on;
    hold off;
    saveas(gcf, fullfile(folder_name, 'Chi2_test_L.tiff')); % Save as .tiff
    if ~display_plots
        close(gcf); % Close the plot if not displaying
    end
end

function [test_results] = perform_noise_verification(num_iterations, num_samples, display_results)
    % Perform verification for noise variance σ_w^2
    % Inputs:
    %   num_iterations - Number of σ_w^2 values to test (e.g., 100)
    %   num_samples    - Number of samples to generate for each value (e.g., 100,000)
    %   display_results - Boolean to decide if results should be printed (true/false)
    % Outputs:
    %   test_results   - Structure containing the results of the mean, variance and f-tests
    

    % Pre-allocate structure to store test results
    test_results(num_iterations).mean_test_sigma_w_2_low = [];
    test_results(num_iterations).f_test_sigma_w_2_low = [];
    test_results(num_iterations).mean_test_sigma_w_2_high = [];
    test_results(num_iterations).f_test_sigma_w_2_high = [];

    % Initialize wait bar
    h = waitbar(0, 'Performing noise variance verification tests');

    % Loop through each iteration to generate and verify
    for iter = 1:num_iterations
        % Update wait bar
        waitbar(iter / num_iterations, h, sprintf('Performing noise variance tests (Iteration %d of %d)', iter, num_iterations));

        % Step 1: Select SNR (High & Low)
        snr_low = (10*rand);
        snr_high = (10 + (20)*rand);
        
        % Step 2: Calculate noise variance σ_w^2
        N = 128;
        sigma_w_2_low = (1/N) * 10^(-snr_low/10);
        sigma_w_2_high = (1/N) * 10^(-snr_high/10);

        % Step 3: Generate 100,000 samples of w using the sigma_w_2_low/high
        num_pilots = 117;
        wn_low = sqrt(sigma_w_2_low) * (randn(num_samples, num_pilots) + 1i*randn(num_samples, num_pilots));
        w_low = wn_low(:,:) ./ sqrt(2);

        wn_high = sqrt(sigma_w_2_high) * (randn(num_samples, num_pilots) + 1i*randn(num_samples, num_pilots));
        w_high = wn_high(:,:) ./ sqrt(2);

        % Step 4: Calculate variance over 100,000 samples
        variance_low = var(w_low, 0, 1);
        variance_high = var(w_high, 0, 1);

        % Step 5: Perform f-test for variance
        f_test_low = variance_low ./ sigma_w_2_low; % f-test compares variance to σ_w^2
        f_test_low(f_test_low < 1) = 1 ./ f_test_low(f_test_low < 1);

        f_test_high = variance_high ./ sigma_w_2_high; % f-test compares variance to σ_w^2
        f_test_high(f_test_high < 1) = 1 ./ f_test_high(f_test_high < 1);

        % Step 6: Calculate mean deviation error of the variance 
        mean_variance_error_low = variance_low - sigma_w_2_low; % e(σ_w^2) = Var(w) - σ_w^2
        mean_variance_error_low = abs(mean_variance_error_low);

        mean_variance_error_high = variance_high - sigma_w_2_high; % e(σ_w^2) = Var(w) - σ_w^2
        mean_variance_error_high = abs(mean_variance_error_high);

        % Step 7: Calculate variance as function of sample size N 
        % Step 7a: Pre-allocate arrays for storing variance decay
        n_values = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000];  % Number of samples N

        variance_decay_low = zeros(length(n_values), 1);  % Store variance of Var_N(w_low)
        variance_decay_high = zeros(length(n_values), 1); % Store variance of Var_N(w_high)
        
        % Step 7b: calculate variance
        for idx = 1:length(n_values)
            N = n_values(idx);  % Select the first N samples

            % Step 7c: Calculate the variance of the first N samples of w_low and w_high
            var_w_low_N = var(w_low(1:N, :), 0, 1);  % Variance of N samples for each pilot
            var_w_high_N = var(w_high(1:N, :), 0, 1);

            % Step 7d: Calculate the variance of the sample variances for w_low and w_high
            variance_decay_low(idx) = var(var_w_low_N);  % Var(Var_N(w_low))
            variance_decay_high(idx) = var(var_w_high_N); % Var(Var_N(w_high))
        end

        % Store variance results in the test_results structure
        test_results(iter).variance_decay_low = variance_decay_low;
        test_results(iter).variance_decay_high = variance_decay_high;

        % Step 8: Store results in structure
        test_results(iter).mean_test_sigma_w_2_low = mean_variance_error_low;
        test_results(iter).f_test_sigma_w_2_low = f_test_low;
        test_results(iter).mean_test_sigma_w_2_high = mean_variance_error_high;
        test_results(iter).f_test_sigma_w_2_high = f_test_high;

        % Optionally display results
        if display_results
            fprintf('Mean Error σ_w^2 low SNR: %s\n', num2str(mean_variance_error_low'));
            fprintf('F-Test σ_w^2 low SNR: %s\n', num2str(f_test_low'));
            fprintf('Mean Error σ_w^2 high SNR: %s\n', num2str(mean_variance_error_high'));
            fprintf('F-Test σ_w^2 high SNR: %s\n', f_test_high);
            fprintf('--------------------------\n');
        end
    end

    % Close the wait bar
    close(h);
end

function plot_noise_test_results(test_results, folder_name, display_plots)
    % Plot the both mean error, f-statistic and variance decay across all iterations
    % Inputs:
    %   test_results  - Structure containing the results of the test
    %   folder_name   - Folder to save the plots
    %   display_plots - Boolean to display the plot (true/false)

    % Ensure save folder exists
    if ~exist(folder_name, 'dir')
        mkdir(folder_name);
    end

    % Get the number of iterations
    num_iterations = length(test_results);

    % ----------- 1. Plot the Mean Error for low SNR ----------- %
    figure;
    hold on;
    for iter = 1:num_iterations
        plot(iter, test_results(iter).mean_test_sigma_w_2_low, 'o-', 'LineWidth', 1.5); % Mean Error per iteration
    end
    title('Mean Variance Error \sigma_w^2 low SNR Across All Iterations');
    xlabel('Iteration');
    ylabel('Mean Variance Error');
    grid on;
    hold off;
    saveas(gcf, fullfile(folder_name, 'mean_error_σ_w^2_low.tiff')); % Save as .tiff
    if ~display_plots
        close(gcf); % Close the plot if not displaying
    end

      % ----------- 2. Plot the Mean Error for high SNR ----------- %
    figure;
    hold on;
    for iter = 1:num_iterations
        plot(iter, test_results(iter).mean_test_sigma_w_2_high, 'o-', 'LineWidth', 1.5); % Mean Error per iteration
    end
    title('Mean Variance Error \sigma_w^2 high SNR Across All Iterations');
    xlabel('Iteration');
    ylabel('Mean Variance Error');
    grid on;
    hold off;
    saveas(gcf, fullfile(folder_name, 'mean_error_σ_w^2_high.tiff')); % Save as .tiff
    if ~display_plots
        close(gcf); % Close the plot if not displaying
    end

    % ----------- 3. Plot the F-Statistic for σ_w^2 low ----------- %
    f_threshold_95 = 1.0105;
    f_threshold_99 = 1.0148;

    figure;
    hold on;
    for iter = 1:num_iterations
        for tap = 1:length(test_results(iter).f_test_sigma_w_2_low)
            f_val = test_results(iter).f_test_sigma_w_2_low(tap);
            if f_val < f_threshold_95
                plot_color = [0.4660 0.6740 0.1880]; % Green for below 95% threshold
            elseif f_val < f_threshold_99
                plot_color = [0.9290 0.6940 0.1250]; % Yellow for between 95% and 99% thresholds
            else
                plot_color = [0.6350 0.0780 0.1840]; % Red for above 99% threshold
            end
            plot(iter, f_val, 'o', 'MarkerEdgeColor', plot_color, 'MarkerFaceColor', plot_color);
        end
    end
    yline(f_threshold_95, '--r', '95% Threshold');
    yline(f_threshold_99, '--b', '99% Threshold');
    title('F-Statistic for \sigma_w^2 low SNR Across All Iterations');
    xlabel('Iteration');
    ylabel('F-Statistic');
    grid on;
    hold off;
    saveas(gcf, fullfile(folder_name, 'f_test_σ_w^2_low.tiff')); % Save as .tiff
    if ~display_plots
        close(gcf); % Close the plot if not displaying
    end

    % ----------- 4. Plot the F-Statistic for σ_w^2 high ----------- %
    f_threshold_95 = 1.0105;
    f_threshold_99 = 1.0148;

    figure;
    hold on;
    for iter = 1:num_iterations
        for tap = 1:length(test_results(iter).f_test_sigma_w_2_high)
            f_val = test_results(iter).f_test_sigma_w_2_high(tap);
            if f_val < f_threshold_95
                plot_color = [0.4660 0.6740 0.1880]; % Green for below 95% threshold
            elseif f_val < f_threshold_99
                plot_color = [0.9290 0.6940 0.1250]; % Yellow for between 95% and 99% thresholds
            else
                plot_color = [0.6350 0.0780 0.1840]; % Red for above 99% threshold
            end
            plot(iter, f_val, 'o', 'MarkerEdgeColor', plot_color, 'MarkerFaceColor', plot_color);
        end
    end
    yline(f_threshold_95, '--r', '95% Threshold');
    yline(f_threshold_99, '--b', '99% Threshold');
    title('F-Statistic for \sigma_w^2 high SNR Across All Iterations');
    xlabel('Iteration');
    ylabel('F-Statistic');
    grid on;
    hold off;
    saveas(gcf, fullfile(folder_name, 'f_test_σ_w^2_high.tiff')); % Save as .tiff
    if ~display_plots
        close(gcf); % Close the plot if not displaying
    end

    % ----------- 5. Plot the Variance Decay for σ_w^2 low & high SNR ----------- %
    n_values = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000];  % X-axis for the plots

    figure;

    % Subplot 1: low SNR
    subplot(1, 2, 1);  % Create a 1x2 grid and use the first subplot
    hold on;

    % Iterate over all test iterations and plot deviation method
    for iter = 1:num_iterations
        plot(n_values, test_results(iter).variance_decay_low(:, 1), '-o', 'LineWidth', 1.5, ...
            'DisplayName', ['Iteration ', num2str(iter)]);
    end

    % Add labels, title, and settings for the deviation method plot
    xlabel('Number of Samples');
    ylabel('Variance of \sigma_w^2 ');
    title('Low SNR case');
    grid on;
    set(gca, 'XScale', 'log');  % Use logarithmic scale for the x-axis
    legend('off');  % Disable the legend to avoid cluttering the plot

    hold off;

    % Subplot 2: Variance Estimation Method
    subplot(1, 2, 2);  % Use the second subplot in the 1x2 grid
    hold on;

    % Iterate over all test iterations and plot estimation method
    for iter = 1:num_iterations
        plot(n_values, test_results(iter).variance_decay_high(:, 1), '-x', 'LineWidth', 1.5, ...
            'DisplayName', ['Iteration ', num2str(iter)]);
    end

    % Add labels, title, and settings for the estimation method plot
    xlabel('Number of Samples');
    ylabel('Variance of \sigma_w^2');
    title('High SNR case');
    grid on;
    set(gca, 'XScale', 'log');  % Use logarithmic scale for the x-axis
    legend('off');  % Disable the legend to avoid cluttering the plot

    hold off;

    % Add overall title to the figure
    sgtitle('Sample Variance Decay - Low & High SNR');

    % Save the figure as a .tiff file
    saveas(gcf, fullfile(folder_name, 'variance_decay_low_high.tiff'));  % Save as .tiff file

    % Close the figure if not displaying
    if ~display_plots
        close(gcf);
    end
end
