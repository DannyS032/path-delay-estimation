import torch
import numpy as np
from torch.utils.data import DataLoader
from data.dataset import ChannelDataset
from model.network import unet, regnet
from torchinterp1d import Interp1d
import os
import time
from tqdm import tqdm
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tabulate import tabulate


def real2complex2dim(x):
    """Convert real tensor to complex for visualization"""
    return torch.complex(x[0:1, :], x[1:2, :])

def interp1d(y, x, x_new):
    """Linear interpolation for the third dimension"""
    N = y.shape[0]
    out = []
    for i in range(N):
        y_new = Interp1d.apply(x[i], y[i], x_new[i])
        out.append(y_new.unsqueeze(0))
    return torch.cat(out, 0).detach()

def estimate_toa_for_sample(models, sample, device):
    """Perform ToA estimation for a single sample"""
    gen_model, regA_model, regB_model = models
    win_siz = 100  # Window size in samples
    win_idx = 128  # Number of points to use for fine estimation
    
    # Move data to device
    cir_l = sample['cir_l'].float().to(device)
    
    # Generate high-res CIR
    cir_h_pred = gen_model(cir_l)
    
    # Coarse ToA estimation
    toa_coarse = regA_model(cir_h_pred)
    toa_coarse = toa_coarse.squeeze(-1)

    # Prepare for fine estimation
    x = 12.5 * torch.arange(0, cir_h_pred.shape[2]).repeat(2, 1).to(device)
    window = torch.arange(-win_siz, win_siz, 2*win_siz/win_idx).repeat(2, 1).to(device)
    x_new = toa_coarse.unsqueeze(-1) + window.unsqueeze(0)
    x = x.repeat(cir_h_pred.shape[0],1,1)
    cir_h_crop = interp1d(cir_h_pred, x, x_new)

    # Fine ToA estimation
    regB_pred = regB_model(cir_h_crop)
    regB_pred = regB_pred.squeeze(-1)
    toa_fine = toa_coarse + regB_pred
    
    return toa_fine.item()

def create_test_directories():
    """Create organized directory structure for L-SNR test results"""
    base_dir = 'test_results_l_snr'
    subdirs = ['tables', 'plots', 'statistics', 'individual_results'  ]
    for dir in [base_dir] + [f"{base_dir}/{subdir}" for subdir in subdirs]:
        os.makedirs(dir, exist_ok=True)
    return base_dir

def process_single_file(file_path, models, device, base_dir):
    """Process a single file and return its results"""
    print(f"\nProcessing file: {file_path}")
    
    try:
        dataset = ChannelDataset(file_path)
    except Exception as e:
        print(f"Error loading dataset from {file_path}: {str(e)}")
        return None
    
    # Initialize results structure
    L_values = []
    SNR_values = []
    file_results = {}
    
    # Process samples
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        current_snr_case = None
        gen_model, regA_model, regB_model = models
        
        for sample in tqdm(dataloader, desc=f"Processing {os.path.basename(file_path)}"):
            l = sample['l'].item()
            snr = sample['snr'].item()
            
            L_values.append(l)
            SNR_values.append(snr)
            
            if l not in file_results:
                file_results[l] = {}
            if snr not in file_results[l]:
                file_results[l][snr] = {'errors': [], 'count': 0}
            
            # Update models if SNR case changes
            snr_case = 'low' if snr <= 10 else 'high'
            if snr_case != current_snr_case:
                gen_model.load_state_dict(torch.load(f'train-gen/gen_model_trained_{snr_case}.w', weights_only=True))
                regA_model.load_state_dict(torch.load(f'train-reg/regA_model_trained_{snr_case}.w', weights_only=True))
                regB_model.load_state_dict(torch.load(f'train-reg/regB_model_trained_{snr_case}.w', weights_only=True))
                current_snr_case = snr_case
                models = (gen_model.eval(), regA_model.eval(), regB_model.eval())
            
            # Process sample
            estimated_toa = estimate_toa_for_sample(models, sample, device)
            error = estimated_toa - sample['toa'].item()
            
            file_results[l][snr]['errors'].append(error)
            file_results[l][snr]['count'] += 1
    
    return {
        'results': file_results,
        'L_values': np.array(L_values),
        'SNR_values': np.array(SNR_values)
    }

def combine_results(all_file_results):
    """Combine results from multiple files"""
    combined_results = {}
    all_L_values = []
    all_SNR_values = []
    
    # Collect all unique L and SNR values
    for result in all_file_results:
        if result is not None:
            all_L_values.extend(result['L_values'])
            all_SNR_values.extend(result['SNR_values'])
    
    unique_L = np.unique(all_L_values)
    unique_SNR = np.unique(all_SNR_values)
    
    # Initialize combined results structure
    for l in unique_L:
        combined_results[l] = {}
        for snr in unique_SNR:
            combined_results[l][snr] = {'errors': [], 'count': 0}
    
    # Combine results from all files
    for result in all_file_results:
        if result is not None:
            for l in result['results']:
                for snr in result['results'][l]:
                    combined_results[l][snr]['errors'].extend(result['results'][l][snr]['errors'])
                    combined_results[l][snr]['count'] += result['results'][l][snr]['count']
    
    return combined_results, unique_L, unique_SNR, np.array(all_L_values), np.array(all_SNR_values)

def calculate_statistics(combined_results, unique_L, unique_SNR):
    """Calculate final statistics from combined results"""
    results_df = pd.DataFrame(index=unique_SNR, columns=unique_L)
    sample_counts_df = pd.DataFrame(index=unique_SNR, columns=unique_L)
    fd_rates_df = pd.DataFrame(index=unique_SNR, columns=['FD_Rate'])
    
    for l in unique_L:
        for snr in unique_SNR:
            errors = combined_results[l][snr]['errors']
            count = combined_results[l][snr]['count']
            
            if count > 0:
                percentile_90 = np.percentile(np.abs(errors), 90)
                results_df.loc[snr, l] = percentile_90
            else:
                results_df.loc[snr, l] = np.nan
            
            sample_counts_df.loc[snr, l] = count

    # Calculate False Detection Rate per SNR
    threshold = 12.5  # 12.5 nsec threshold as specified
    for snr in unique_SNR:
        total_errors = []
        # Collect all errors for this SNR across all L values
        for l in unique_L:
            total_errors.extend(combined_results[l][snr]['errors'])
        
        if total_errors:
            # Calculate FD rate: number of errors > threshold divided by total samples
            fd_count = sum(abs(error) > threshold for error in total_errors)
            fd_rate = fd_count / len(total_errors)
            fd_rates_df.loc[snr, 'FD_Rate'] = fd_rate
        else:
            fd_rates_df.loc[snr, 'FD_Rate'] = np.nan
    
    # Sort by SNR values descending
    results_df = results_df.sort_index(ascending=False)
    sample_counts_df = sample_counts_df.sort_index(ascending=False)
    fd_rates_df = fd_rates_df.sort_index(ascending=False)
    
    return results_df, sample_counts_df, fd_rates_df

def analyze_data_distribution(dataset):
    """Analyze the distribution of L and SNR in the dataset"""
    L_values = []
    SNR_values = []
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for sample in tqdm(dataloader, desc="Analyzing data distribution"):
        L_values.append(sample['l'].item())
        SNR_values.append(sample['snr'].item())
    
    L_values = np.array(L_values)
    SNR_values = np.array(SNR_values)
    
    # Get unique L values (should be integers 3-15)
    unique_L = np.unique(L_values)
    
    # Get unique SNR values (30:-3:-5 from MATLAB code)
    unique_SNR = np.unique(SNR_values)
    
    return unique_L, unique_SNR, L_values, SNR_values

def test_l_snr(test_files):
    """Process multiple test files separately and combine results"""
    base_dir = create_test_directories()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    gen_model = unet().to(device)
    regA_model = regnet(256).to(device)
    regB_model = regnet(128).to(device)
    models = (gen_model, regA_model, regB_model)
    
    # Process each file separately
    all_file_results = []
    for file_path in test_files:
        file_results = process_single_file(file_path, models, device, base_dir)
        all_file_results.append(file_results)
        
        # Save individual file results
        if file_results is not None:
            file_name = os.path.basename(file_path).split('.')[0]
            np.save(f'{base_dir}/individual_results/{file_name}_results.npy', file_results)
    
    # Combine results from all files
    print("\nCombining results from all files...")
    combined_results, unique_L, unique_SNR, all_L_values, all_SNR_values = combine_results(all_file_results)
    
    # Calculate final statistics
    print("Calculating final statistics...")
    results_df, sample_counts_df, fd_rates_df = calculate_statistics(combined_results, unique_L, unique_SNR)
    
    # Save all results
    results_df.to_csv(f'{base_dir}/tables/NN_final_results.csv')
    sample_counts_df.to_csv(f'{base_dir}/tables/sample_counts.csv')
    fd_rates_df.to_csv(f'{base_dir}/tables/false_detection_rates.csv')
    
    save_visualizations(results_df, unique_L, unique_SNR, base_dir)
    save_fd_visualizations(fd_rates_df, base_dir)
    save_distribution_analysis(all_L_values, all_SNR_values, unique_L, unique_SNR, base_dir)
    save_results_table(results_df, fd_rates_df, base_dir)
    
    print(f"\nResults saved in {base_dir}")

def save_visualizations(results_df, unique_L, unique_SNR, base_dir):
    """Generate and save visualization plots"""
    plot_data = results_df.values.astype(float)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(plot_data, aspect='auto', cmap='viridis')
    plt.colorbar(label='90th Percentile Error')
    plt.xlabel('Number of Taps (L)')
    plt.ylabel('SNR (dB)')
    plt.title('90th Percentile ToA Estimation Error')
    plt.xticks(range(len(unique_L)), unique_L)
    plt.yticks(range(len(unique_SNR)), [f'{snr:.1f}' for snr in unique_SNR[::-1]])
    plt.savefig(f'{base_dir}/plots/error_heatmap.png')
    plt.close()

def save_fd_visualizations(fd_rates_df, base_dir):
    """Generate and save FD rate visualization plot"""
    plt.figure(figsize=(10, 6))
    snr_values = fd_rates_df.index
    fd_rates = fd_rates_df['FD_Rate']
    
    plt.plot(snr_values, fd_rates, 'b-o')
    plt.grid(True)
    plt.xlabel('SNR (dB)')
    plt.ylabel('False Detection Rate')
    plt.title('False Detection Rate vs SNR')
    plt.xticks(snr_values)
    
    # Add percentage labels on each point
    for x, y in zip(snr_values, fd_rates):
        plt.annotate(f'{y:.2%}', 
                    (x, y),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')
    
    plt.savefig(f'{base_dir}/plots/false_detection_rates.png')
    plt.close()

def save_distribution_analysis(L_values, SNR_values, unique_L, unique_SNR, base_dir):
    """Save distribution analysis to file"""
    with open(f'{base_dir}/statistics/data_distribution.txt', 'w') as f:
        f.write("Sample Distribution Analysis\n")
        f.write("===========================\n\n")
        f.write(f"Total samples: {len(L_values)}\n\n")
        
        f.write("L value distribution:\n")
        for l in unique_L:
            count = np.sum(L_values == l)
            f.write(f"L={l}: {count} samples ({count/len(L_values)*100:.1f}%)\n")
        
        f.write("\nSNR distribution:\n")
        for snr in unique_SNR:
            count = np.sum(SNR_values == snr)
            f.write(f"SNR={snr:.1f}dB: {count} samples ({count/len(SNR_values)*100:.1f}%)\n")

def save_results_table(results_df, fd_rates_df, base_dir):
    """Save results in tabulated format"""
    table = []
    headers = ['SNR(dB)'] + [f'L={l}' for l in results_df.columns] + ['FD_Rate']
    
    for snr_idx in results_df.index:
        row = [f'{snr_idx:.1f}']
        for l in results_df.columns:
            value = results_df.loc[snr_idx, l]
            row.append(f'{value:.3f}' if not np.isnan(value) else 'N/A')
        # Add FD rate
        fd_rate = fd_rates_df.loc[snr_idx, 'FD_Rate']
        row.append(f'{fd_rate:.3%}' if not np.isnan(fd_rate) else 'N/A')
        table.append(row)
    
    # Save as text file with fixed-width format
    with open(f'{base_dir}/tables/L_SNR_results_table.txt', 'w') as f:
        f.write(tabulate(table, headers=headers, tablefmt='grid'))
    
    # Save as CSV
    with open(f'{base_dir}/tables/L_SNR_results_table.csv', 'w') as f:
        f.write(','.join(headers) + '\n')
        for row in table:
            f.write(','.join(str(x) for x in row) + '\n')

if __name__ == '__main__':
    # List of test files
    test_files = [
        'data/test_data_L_SNR_1.h5',
        'data/test_data_L_SNR_2.h5'
    ]
    test_l_snr(test_files)