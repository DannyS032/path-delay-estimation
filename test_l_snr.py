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
    subdirs = ['tables', 'plots', 'statistics']
    for dir in [base_dir] + [f"{base_dir}/{subdir}" for subdir in subdirs]:
        os.makedirs(dir, exist_ok=True)
    return base_dir

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
    
    # Create SNR bins (let's use 1dB steps)
    snr_min = np.floor(min(SNR_values))
    snr_max = np.ceil(max(SNR_values))
    snr_bins = np.arange(snr_min, snr_max + 1, 1.0)
    
    return unique_L, snr_bins, L_values, SNR_values

def test_l_snr(test_file):
    """
    L-SNR testing adapted for existing data distribution
    Args:
        test_file: Path to test data file
    """
    base_dir = create_test_directories()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    gen_model = unet().to(device)
    regA_model = regnet(256).to(device)
    regB_model = regnet(128).to(device)
    
    # Load weights based on SNR case
    snr_case = 'high' if 'high' in test_file else 'low'
    gen_model.load_state_dict(torch.load(f'train-gen-1m/gen_model_trained_{snr_case}.w', weights_only=True))
    regA_model.load_state_dict(torch.load(f'train-reg-1m/regA_model_trained_{snr_case}.w', weights_only=True))
    regB_model.load_state_dict(torch.load(f'train-reg-1m/regB_model_trained_{snr_case}.w', weights_only=True))
    
    models = (gen_model.eval(), regA_model.eval(), regB_model.eval())
    
    # Load dataset and analyze distribution
    dataset = ChannelDataset(test_file)
    unique_L, snr_bins, L_values, SNR_values = analyze_data_distribution(dataset)
    
    # Create storage for results
    results = {}
    for l in unique_L:
        results[l] = {snr: {'errors': [], 'count': 0} 
                     for snr in snr_bins[:-1]}
    
    # Testing loop
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        for sample in tqdm(dataloader, desc="Processing samples"):
            # Get L and SNR values
            l = sample['l'].item()
            snr = sample['snr'].item()
            
            # Find SNR bin
            snr_bin = snr_bins[np.digitize(snr, snr_bins) - 1]
            
            # Process sample through models and get error
            # (Replace with your actual estimation code)
            estimated_toa = estimate_toa_for_sample(models, sample, device)
            error = estimated_toa - sample['toa'].item()
            
            # Store result
            results[l][snr_bin]['errors'].append(error)
            results[l][snr_bin]['count'] += 1
    
    # Create results DataFrame
    results_df = pd.DataFrame(index=snr_bins[:-1], columns=unique_L)
    sample_counts_df = pd.DataFrame(index=snr_bins[:-1], columns=unique_L)
    
    # Calculate 90th percentile errors and store sample counts
    for l in unique_L:
        for snr in snr_bins[:-1]:
            errors = results[l][snr]['errors']
            count = results[l][snr]['count']
            
            if count >= 100:  # Minimum sample threshold
                percentile_90 = np.percentile(np.abs(errors), 90)
                results_df.loc[snr, l] = percentile_90
            else:
                results_df.loc[snr, l] = np.nan
            
            sample_counts_df.loc[snr, l] = count
    
    # Save results
    results_df.to_csv(f'{base_dir}/tables/percentile_90_results_{snr_case}.csv')
    sample_counts_df.to_csv(f'{base_dir}/tables/sample_counts_{snr_case}.csv')

    plot_data = results_df.values.astype(float)
    
    # Generate heatmap visualization
    plt.figure(figsize=(12, 8))
    plt.imshow(plot_data, aspect='auto', cmap='viridis')
    plt.colorbar(label='90th Percentile Error')
    plt.xlabel('Number of Taps (L)')
    plt.ylabel('SNR (dB)')
    plt.title(f'90th Percentile ToA Estimation Error ({snr_case.upper()} SNR)')
    plt.xticks(range(len(unique_L)), unique_L)
    plt.yticks(range(len(snr_bins[:-1])), [f'{snr:.1f}' for snr in snr_bins[:-1]])
    plt.savefig(f'{base_dir}/plots/error_heatmap_{snr_case}.png')
    plt.close()
    
    # Save distribution analysis
    with open(f'{base_dir}/statistics/data_distribution_{snr_case}.txt', 'w') as f:
        f.write("Sample Distribution Analysis\n")
        f.write("===========================\n\n")
        f.write(f"Total samples: {len(L_values)}\n\n")
        f.write("L value distribution:\n")
        for l in unique_L:
            count = np.sum(L_values == l)
            f.write(f"L={l}: {count} samples ({count/len(L_values)*100:.1f}%)\n")
        f.write("\nSNR distribution:\n")
        for i in range(len(snr_bins)-1):
            mask = (SNR_values >= snr_bins[i]) & (SNR_values < snr_bins[i+1])
            count = np.sum(mask)
            f.write(f"SNR {snr_bins[i]:.1f}-{snr_bins[i+1]:.1f}dB: {count} samples ({count/len(SNR_values)*100:.1f}%)\n")

def save_results_table(results_df, base_dir):
    """
    Save results in the requested table format
    Args:
        results_df: DataFrame with results
        base_dir: Directory to save results
    """
    # Format the table
    # First column will be SNR values
    table = []
    headers = ['SNR(dB)'] + [f'L={l}' for l in results_df.columns]
    
    # Add each row (SNR value) with its corresponding 90th percentile errors
    for snr_idx in results_df.index:
        row = [f'{snr_idx:.1f}']
        for l in results_df.columns:
            value = results_df.loc[snr_idx, l]
            row.append(f'{value:.3f}' if not np.isnan(value) else 'N/A')
        table.append(row)
    
    # Save as text file with fixed-width format
    with open(f'{base_dir}/tables/L_SNR_results_table.txt', 'w') as f:
        f.write(tabulate(table, headers=headers, tablefmt='grid'))
    
    # Save as CSV for easier data processing
    with open(f'{base_dir}/tables/L_SNR_results_table.csv', 'w') as f:
        f.write(','.join(headers) + '\n')
        for row in table:
            f.write(','.join(str(x) for x in row) + '\n')

if __name__ == '__main__':
    # Create a dataset that combines both high and low SNR data
    dataset_high = ChannelDataset('data/test_data_high.h5')
    dataset_low = ChannelDataset('data/test_data_low.h5')
    
    # First, analyze distribution of both datasets
    unique_L_high, snr_bins_high, L_values_high, SNR_values_high = analyze_data_distribution(dataset_high)
    unique_L_low, snr_bins_low, L_values_low, SNR_values_low = analyze_data_distribution(dataset_low)
    
    # Combine the SNR ranges
    snr_min = min(np.min(SNR_values_low), np.min(SNR_values_high))
    snr_max = max(np.max(SNR_values_low), np.max(SNR_values_high))
    
    print(f"Full SNR range: {snr_min:.1f} to {snr_max:.1f} dB")
    
    # Process both datasets
    for test_file in ['data/test_data_high.h5', 'data/test_data_low.h5']:
        print(f"\nProcessing {test_file}...")
        
        # For high SNR file, use high SNR model
        if 'high' in test_file:
            model_suffix = 'high'
        else:
            model_suffix = 'low'
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models with appropriate weights
        gen_model = unet().to(device)
        regA_model = regnet(256).to(device)
        regB_model = regnet(128).to(device)
        
        gen_model.load_state_dict(torch.load(f'train-gen-1m/gen_model_trained_{model_suffix}.w', weights_only=True))
        regA_model.load_state_dict(torch.load(f'train-reg-1m/regA_model_trained_{model_suffix}.w', weights_only=True))
        regB_model.load_state_dict(torch.load(f'train-reg-1m/regB_model_trained_{model_suffix}.w', weights_only=True))
        
        models = (gen_model.eval(), regA_model.eval(), regB_model.eval())
        
        # Test file
        test_l_snr(test_file)

    # Now combine the results from both CSV files
    base_dir = 'test_results_l_snr'
    high_results = pd.read_csv(f'{base_dir}/tables/percentile_90_results_high.csv', index_col=0)
    low_results = pd.read_csv(f'{base_dir}/tables/percentile_90_results_low.csv', index_col=0)
    
    # Combine and sort by SNR
    combined_results = pd.concat([high_results, low_results])
    combined_results = combined_results.sort_index()
    
    # Remove duplicate SNR values if any (take the better result)
    combined_results = combined_results.groupby(level=0).min()
    
    # Save combined results
    combined_results.to_csv(f'{base_dir}/tables/percentile_90_results_combined.csv')
    
    # Generate combined heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(combined_results.values, aspect='auto', cmap='viridis')
    plt.colorbar(label='90th Percentile Error')
    plt.xlabel('Number of Taps (L)')
    plt.ylabel('SNR (dB)')
    plt.title('90th Percentile ToA Estimation Error (Combined SNR)')
    plt.xticks(range(len(combined_results.columns)), combined_results.columns)
    plt.yticks(range(len(combined_results.index)), [f'{snr:.1f}' for snr in combined_results.index])
    plt.savefig(f'{base_dir}/plots/error_heatmap_combined.png')
    plt.close()
    
    print("\nCombined results saved in test_results_l_snr/")

    # Save the final results table
    save_results_table(combined_results, base_dir)
    print("\nResults table saved in test_results_l_snr/tables/L_SNR_results_table.txt")