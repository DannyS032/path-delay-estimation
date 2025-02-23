clc, clear;

% Read the CSV files
music_data = readtable('test_results_l_snr/music/MUSIC_final_results.csv');
nn_data = readtable('test_results_l_snr/tables/NN_final_results.csv');

music_fd_rates = readtable('test_results_l_snr/music/MUSIC_fd_rates.csv');
nn_fd_rates = readtable('test_results_l_snr/tables/false_detection_rates.csv');

% Convert tables to matrices (excluding the first column which contains labels)
music_matrix = table2array(music_data(:, 2:end));
nn_matrix = table2array(nn_data(2:end, 2:end));

music_fd_matrix = table2array(music_fd_rates(:, 2:end));
music_fd_matrix = music_fd_matrix .* 100;
nn_fd_matrix = table2array(nn_fd_rates(:, 2:end));
nn_fd_matrix = nn_fd_matrix .* 100;

% Calculate difference (MUSIC - NN)
diff_matrix = music_matrix - nn_matrix;
diff_fd_matrix = music_fd_matrix - nn_fd_matrix;

% Calculate Error percentage improvement
error_percent_matrix = (abs(diff_matrix) ./ music_matrix) * 100;
error_percent_fd_matrix = (abs(diff_fd_matrix) ./ music_fd_matrix) * 100;

% Create table with the same column names as input
column_names = music_data.Properties.VariableNames(2:end);
diff_table = array2table(diff_matrix, 'VariableNames', column_names);
error_percent_table = array2table(error_percent_matrix, 'VariableNames', column_names);

% Add SNR column from original data
diff_table = addvars(diff_table, music_data.Row, 'Before', 1, 'NewVariableNames', {'SNR'});
error_percent_table = addvars(error_percent_table, music_data.Row, 'Before', 1, 'NewVariableNames', {'SNR'});

% Create combined table for false detection rates
fd_table = array2table([music_fd_matrix nn_fd_matrix], 'VariableNames', {'MUSIC_FD_Rate', 'NN_FD_Rate'});

% Add SNR column from original data
fd_table = addvars(fd_table, music_data.Row, 'Before', 1, 'NewVariableNames', {'SNR'});

% Write results to CSV
writetable(diff_table, 'final_analysis/difference_results.csv');
writetable(error_percent_table, 'final_analysis/error_percent_results.csv')
writetable(fd_table, 'final_analysis/combined_fd_rates.csv');


% Display results
disp('Difference Table (MUSIC - NN):');
disp(diff_table);

% Plot heatmap of differences
figure;
imagesc(diff_matrix);
colorbar;
title('Difference between NN and MUSIC (MUSIC - NN)');
xlabel('L value');
ylabel('SNR (dB)');
set(gca, 'XTick', 1:length(column_names), 'XTickLabel', column_names);
snr_values = music_data.Row;
set(gca, 'YTick', 1:length(snr_values), 'YTickLabel', snr_values);

nn_table = array2table(nn_matrix, 'VariableNames', column_names);
nn_table = addvars(nn_table, music_data.Row, 'Before', 1, 'NewVariableNames', {'SNR'});

% Create heatmap visualizations
L_values = 3:15;
SNR_values = 30:-3:-5;
% Create table with row and column labels
L_labels = arrayfun(@(x) sprintf('L=%d', x), L_values, 'UniformOutput', false);
SNR_labels = arrayfun(@(x) sprintf('SNR=%d', x), SNR_values, 'UniformOutput', false);

figure;
sgtitle('90th Percentile MAE for ToA Estimation');

% Calculate the overall min and max values
min_val = min(min(min(nn_matrix)), min(min(music_matrix)));
max_val = max(max(max(nn_matrix)), max(max(music_matrix)));

% First subplot
subplot(1,2,1);
imagesc(nn_matrix);
colorbar;
title('Neural Network algorithm');
xlabel('Number of Taps (L)');
ylabel('SNR (dB)');
xticks(1:length(L_values));
xticklabels(L_labels);
yticks(1:length(SNR_values));
yticklabels(SNR_labels);
clim([min_val max_val]);  % Set color limits
colormap('jet');

% Second subplot
subplot(1,2,2);
imagesc(music_matrix);
colorbar;
title('MUSIC algorithm');
xlabel('Number of Taps (L)');
ylabel('SNR (dB)');
xticks(1:length(L_values));
xticklabels(L_labels);
yticks(1:length(SNR_values));
yticklabels(SNR_labels);
clim([min_val max_val]);  % Set color limits
colormap('jet');

savefig('final_analysis/toa_error_heatmap(nn_vs_music)');