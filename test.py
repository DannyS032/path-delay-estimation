import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
from data.dataset import ChannelDataset
from model.network import unet, regnet
from model.loss import real2complex
from torchinterp1d import Interp1d
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import random

def create_results_directories():
    """Create organized directory structure for test results"""
    base_dir = 'test_results'
    directories = [
        f'{base_dir}/interpolation',
        f'{base_dir}/cyclic_shift',
        f'{base_dir}/comparative_analysis'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    return base_dir

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

def plot_sample(cir_l, cir_h, cir_h_pred, cir_h_crop, x_new, toa, toa_coarse, toa_fine, snr_case, sample_idx, method_dir):
    """Create visualization plots for a single sample"""
    x_plot = 12.5 * torch.arange(0, 256)
    x_new_plot = x_new[0,:].cpu().numpy()

    # Convert to numpy and get absolute values
    cir_l_plot = np.abs(real2complex2dim(cir_l).cpu().numpy().squeeze())
    cir_h_plot = np.abs(real2complex2dim(cir_h).cpu().numpy().squeeze())
    cir_h_pred_plot = np.abs(real2complex2dim(cir_h_pred).cpu().numpy().squeeze())
    cir_h_crop_plot = np.abs(real2complex2dim(cir_h_crop).cpu().numpy().squeeze())

    plt.figure(figsize=(16,8))
    
    # Input Plot
    plt.subplot(1, 3, 1)
    plt.plot(x_plot[:70], cir_l_plot[:70], label="Low-res. CIR")
    plt.plot(x_plot[:70], cir_h_plot[:70], label="Ground Truth High-res. CIR")
    plt.axvline(toa, linewidth=2, linestyle="--", color='black', label=f'ToA {toa:.1f}')
    plt.axvline(toa_coarse, linewidth=2, linestyle="--", color='red', label=f'Coarse ToA {toa_coarse:.1f}')
    plt.axvline(toa_fine, linewidth=2, linestyle="--", color='green', label=f'Fine ToA {toa_fine:.1f}')
    plt.title("Noisy low-res. CIR")
    plt.legend()
    
    # Generated CIR Plot
    plt.subplot(1, 3, 2)
    plt.plot(x_plot[:70], cir_h_pred_plot[:70], label="Generated High-res. CIR")
    plt.plot(x_plot[:70], cir_h_plot[:70], label="Ground Truth High-res. CIR")
    plt.axvline(toa, linewidth=2, linestyle="--", color='black', label=f'ToA {toa:.1f}')
    plt.axvline(toa_coarse, linewidth=2, linestyle="--", color='red', label=f'Coarse ToA {toa_coarse:.1f}')
    plt.axvline(toa_fine, linewidth=2, linestyle="--", color='green', label=f'Fine ToA {toa_fine:.1f}')
    plt.title("Generated noiseless high-res. CIR")
    plt.legend()

    # Cropped CIR Plot
    plt.subplot(1, 3, 3)
    plt.plot(x_new_plot, cir_h_crop_plot, label="Cropped CIR (High)", color='g')
    plt.axvline(toa, linewidth=2, linestyle="--", color='black', label=f'ToA {toa:.1f}')
    plt.axvline(toa_coarse, linewidth=2, linestyle="--", color='red', label=f'Coarse ToA {toa_coarse:.1f}')
    plt.axvline(toa_fine, linewidth=2, linestyle="--", color='green', label=f'Fine ToA {toa_fine:.1f}')
    plt.title("Cropped high-res. CIR around coarse ToA")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{method_dir}/CIR_ToA_plots_{snr_case}_sample_{sample_idx}.png')
    plt.close()

def analyze_performance_comparison(results_interp, results_cycshift, snr_case, base_dir):
    """Compare performance metrics between interpolation and cyclic shift methods"""
    comparison_dir = f'{base_dir}/comparative_analysis'
    
    plt.figure(figsize=(15, 10))
    
    # Error Distribution Comparison
    plt.subplot(2, 2, 1)
    plt.hist(results_interp['errors'], bins=50, density=True, alpha=0.6, label='Interpolation', color='blue')
    plt.hist(results_cycshift['errors'], bins=50, density=True, alpha=0.6, label='Cyclic Shift', color='red')
    plt.xlabel('ToA Estimation Error')
    plt.ylabel('Density')
    plt.title('Error Distribution Comparison')
    plt.legend()
    plt.grid(True)

    # Cumulative Error Distribution
    plt.subplot(2, 2, 2)
    for errors, label, color in [(results_interp['errors'], 'Interpolation', 'blue'),
                                (results_cycshift['errors'], 'Cyclic Shift', 'red')]:
        sorted_errors = np.sort(np.abs(errors))
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        plt.plot(sorted_errors, cumulative, label=label, color=color)
    plt.xlabel('Absolute Error')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Error Distribution')
    plt.legend()
    plt.grid(True)

    # Error vs. Ground Truth ToA
    plt.subplot(2, 2, 3)
    plt.scatter(results_interp['true_toa'], results_interp['errors'], 
               alpha=0.3, label='Interpolation', color='blue', s=10)
    plt.scatter(results_cycshift['true_toa'], results_cycshift['errors'], 
               alpha=0.3, label='Cyclic Shift', color='red', s=10)
    plt.xlabel('Ground Truth ToA')
    plt.ylabel('Estimation Error')
    plt.title('Error vs. Ground Truth ToA')
    plt.legend()
    plt.grid(True)

    # Box Plot Comparison
    plt.subplot(2, 2, 4)
    plt.boxplot([results_interp['errors'], results_cycshift['errors']], 
                labels=['Interpolation', 'Cyclic Shift'])
    plt.ylabel('Error Distribution')
    plt.title('Error Distribution Box Plot')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{comparison_dir}/performance_comparison_{snr_case}.png')
    plt.close()

    # Save detailed comparative metrics
    with open(f'{comparison_dir}/comparative_metrics_{snr_case}.txt', 'w') as f:
        methods = {'Interpolation': results_interp, 'Cyclic Shift': results_cycshift}
        for method, results in methods.items():
            errors = results['errors']
            f.write(f"\n{method} Method Metrics:\n")
            f.write(f"RMSE: {np.sqrt(np.mean(errors**2)):.4f}\n")
            f.write(f"Mean Error: {np.mean(errors):.4f}\n")
            f.write(f"Median Error: {np.median(errors):.4f}\n")
            f.write(f"Std Dev: {np.std(errors):.4f}\n")
            f.write(f"Max Absolute Error: {np.max(np.abs(errors)):.4f}\n")
            f.write(f"Samples within ±0.5: {np.mean(np.abs(errors) <= 0.5)*100:.2f}%\n")
            f.write(f"Samples within ±1.0: {np.mean(np.abs(errors) <= 1.0)*100:.2f}%\n")
            f.write(f"Samples within ±2.0: {np.mean(np.abs(errors) <= 2.0)*100:.2f}%\n")
            f.write(f"Positive Error Bias: {np.mean(errors > 0)*100:.2f}%\n")
            f.write(f"Negative Error Bias: {np.mean(errors < 0)*100:.2f}%\n")
            f.write(f"95th Percentile Error: {np.percentile(np.abs(errors), 95):.4f}\n")
            f.write(f"99th Percentile Error: {np.percentile(np.abs(errors), 99):.4f}\n")

def test_model(test_file, snr_case, num_plots=20):
    """Run inference testing using interpolation method"""
    base_dir = create_results_directories()
    method_dir = f'{base_dir}/interpolation'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    win_siz = 100  # Window size in samples
    win_idx = 128  # Number of points to use for fine estimation

    # Initialize models
    gen_model = unet().to(device)
    regA_model = regnet(256).to(device)
    regB_model = regnet(win_idx).to(device)

    # Load trained weights
    gen_model.load_state_dict(torch.load(f'train-gen-1m/gen_model_trained_{snr_case}.w'))
    regA_model.load_state_dict(torch.load(f'train-reg-1m/regA_model_trained_{snr_case}.w'))
    regB_model.load_state_dict(torch.load(f'train-reg-1m/regB_model_trained_{snr_case}.w'))

    # Set models to evaluation mode
    gen_model.eval()
    regA_model.eval()
    regB_model.eval()

    # Load test dataset
    dataset = ChannelDataset(test_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Initialize results storage
    results = {
        'errors': [],
        'true_toa': [],
        'coarse_errors': [],
        'fine_errors': [],
        'processing_times': []
    }
    plot_indices = random.sample(range(len(dataset)), num_plots)

    with torch.no_grad():
        for i, frame in enumerate(tqdm(dataloader, desc="Testing with Interpolation")):
            start_time = time.time()
            
            # Move data to device
            cir_l = frame['cir_l'].float().to(device)
            cir_h = frame['cir_h'].float().to(device)
            toa = frame['toa'].float().to(device)

            # Generate high-res CIR
            cir_h_pred = gen_model(cir_l)
            
            # Coarse ToA estimation
            toa_coarse = regA_model(cir_h_pred)
            toa_coarse = toa_coarse.squeeze(-1)

            # Prepare for fine estimation
            x = 12.5 * torch.arange(0, cir_h.shape[2]).repeat(2, 1).to(device)
            window = torch.arange(-win_siz, win_siz, 2*win_siz/win_idx).repeat(2, 1).to(device)
            x_new = toa_coarse.unsqueeze(-1) + window.unsqueeze(0)
            x = x.repeat(cir_h_pred.shape[0],1,1)
            cir_h_crop = interp1d(cir_h_pred, x, x_new)

            # Fine ToA estimation
            regB_pred = regB_model(cir_h_crop)
            regB_pred = regB_pred.squeeze(-1)
            toa_fine = toa_coarse + regB_pred

            # Store results
            error = toa_fine - toa
            results['errors'].append(error.item())
            results['true_toa'].append(toa.item())
            results['coarse_errors'].append((toa_coarse - toa).item())
            results['fine_errors'].append(error.item())
            results['processing_times'].append(time.time() - start_time)

            # Generate plots for selected samples
            if i in plot_indices:
                plot_sample(cir_l[0], cir_h[0], cir_h_pred[0], cir_h_crop[0],
                          x_new[0], toa[0].item(), toa_coarse[0].item(),
                          toa_fine[0].item(), snr_case, i, method_dir)

    # Convert lists to numpy arrays
    for key in results:
        results[key] = np.array(results[key])

    # Calculate and save metrics
    rmse = np.sqrt(np.mean(results['errors']**2))

    # Plot error histogram
    plt.figure(figsize=(10, 6))
    plt.hist(results['errors'], bins=50, density=True, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    plt.xlabel('ToA Estimation Error (True - Estimated)')
    plt.ylabel('Density')
    plt.title(f'ToA Estimation Error Distribution ({snr_case} SNR, Interpolation)\nRMSE: {rmse:.2f}')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{method_dir}/error_distribution_{snr_case}.png')
    plt.close()

    # Save metrics
    with open(f'{method_dir}/metrics_{snr_case}.txt', 'w') as f:
        f.write(f'RMSE: {rmse:.4f}\n')
        f.write(f'Mean Error: {np.mean(results["errors"]):.4f}\n')
        f.write(f'Median Error: {np.median(results["errors"]):.4f}\n')
        f.write(f'Std Dev: {np.std(results["errors"]):.4f}\n')
        f.write(f'Max Error: {np.max(np.abs(results["errors"])):.4f}\n')
        f.write(f'Mean Processing Time: {np.mean(results["processing_times"])*1000:.2f} ms\n')
        f.write(f'Samples within ±1: {np.mean(np.abs(results["errors"]) <= 1)*100:.2f}%\n')
        f.write(f'Samples within ±2: {np.mean(np.abs(results["errors"]) <= 2)*100:.2f}%\n')

    return results

def test_model_cycshift_window(test_file, snr_case, num_plots=20):
    """Run inference testing using cyclic shifting instead of interpolation"""
    base_dir = create_results_directories()
    method_dir = f'{base_dir}/cyclic_shift'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    win_siz = 100  # Window size in samples
    win_idx = 128  # Number of points to use for fine estimation

    # Initialize models
    gen_model = unet().to(device)
    regA_model = regnet(256).to(device)
    regB_model = regnet(win_idx).to(device)

    # Load trained weights
    gen_model.load_state_dict(torch.load(f'train-gen-1m/gen_model_trained_{snr_case}.w'))
    regA_model.load_state_dict(torch.load(f'train-reg-1m/regA_model_trained_{snr_case}.w'))
    regB_model.load_state_dict(torch.load(f'train-reg-1m/regB_model_trained_{snr_case}.w'))

    # Set models to evaluation mode
    gen_model.eval()
    regA_model.eval()
    regB_model.eval()

    # Load test dataset
    dataset = ChannelDataset(test_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Initialize results storage
    results = {
        'errors': [],
        'true_toa': [],
        'coarse_errors': [],
        'fine_errors': [],
        'processing_times': []
    }
    plot_indices = random.sample(range(len(dataset)), num_plots)

    with torch.no_grad():
        for i, frame in enumerate(tqdm(dataloader, desc="Testing with Cyclic Shift")):
            start_time = time.time()
            
            # Move data to device
            cir_l = frame['cir_l'].float().to(device)
            cir_h = frame['cir_h'].float().to(device)
            toa = frame['toa'].float().to(device)

            # Generate high-res CIR
            cir_h_pred = gen_model(cir_l)
            
            # Coarse ToA estimation
            toa_coarse = regA_model(cir_h_pred)
            toa_coarse = toa_coarse.squeeze(-1)

            # Calculate required shift based on coarse ToA
            center_idx = torch.round(toa_coarse / 12.5).long()  # Convert time to sample index
            
            # Prepare cyclic shifted window
            N = cir_h_pred.shape[2]
            window_start = center_idx - win_siz//2
            
            # Create evenly spaced indices for the window
            step = win_siz / win_idx
            indices = torch.arange(0, win_idx, device=device)
            indices = indices.unsqueeze(0) * step
            indices = indices + window_start.unsqueeze(-1)
            
            # Apply cyclic indexing
            cyclic_indices = indices % N
            cyclic_indices = cyclic_indices.long()
            
            # Apply cyclic shift and extract window
            cir_h_crop = torch.zeros((cir_h_pred.shape[0], 2, win_idx), device=device)
            for b in range(cir_h_pred.shape[0]):
                for c in range(2):  # For both real and imaginary channels
                    cir_h_crop[b, c, :] = cir_h_pred[b, c, cyclic_indices[b]]

            # Fine ToA estimation
            regB_pred = regB_model(cir_h_crop)
            regB_pred = regB_pred.squeeze(-1)
            toa_fine = toa_coarse + regB_pred

            # Store results
            error = toa_fine - toa
            results['errors'].append(error.item())
            results['true_toa'].append(toa.item())
            results['coarse_errors'].append((toa_coarse - toa).item())
            results['fine_errors'].append(error.item())
            results['processing_times'].append(time.time() - start_time)

            # Generate plots for selected samples
            if i in plot_indices:
                x_new = torch.arange(win_idx, device=device) * (2 * win_siz / win_idx) - win_siz
                x_new = x_new.unsqueeze(0) + toa_coarse.unsqueeze(-1)
                plot_sample(cir_l[0], cir_h[0], cir_h_pred[0], cir_h_crop[0],
                          x_new, toa[0].item(), toa_coarse[0].item(),
                          toa_fine[0].item(), snr_case, i, method_dir)

    # Convert lists to numpy arrays
    for key in results:
        results[key] = np.array(results[key])

    # Calculate and save metrics
    rmse = np.sqrt(np.mean(results['errors']**2))

    # Plot error histogram
    plt.figure(figsize=(10, 6))
    plt.hist(results['errors'], bins=50, density=True, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    plt.xlabel('ToA Estimation Error (True - Estimated)')
    plt.ylabel('Density')
    plt.title(f'ToA Estimation Error Distribution ({snr_case} SNR, Cyclic Shift)\nRMSE: {rmse:.2f}')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{method_dir}/error_distribution_{snr_case}.png')
    plt.close()

    # Save metrics
    with open(f'{method_dir}/metrics_{snr_case}.txt', 'w') as f:
        f.write(f'RMSE: {rmse:.4f}\n')
        f.write(f'Mean Error: {np.mean(results["errors"]):.4f}\n')
        f.write(f'Median Error: {np.median(results["errors"]):.4f}\n')
        f.write(f'Std Dev: {np.std(results["errors"]):.4f}\n')
        f.write(f'Max Error: {np.max(np.abs(results["errors"])):.4f}\n')
        f.write(f'Mean Processing Time: {np.mean(results["processing_times"])*1000:.2f} ms\n')
        f.write(f'Samples within ±1: {np.mean(np.abs(results["errors"]) <= 1)*100:.2f}%\n')
        f.write(f'Samples within ±2: {np.mean(np.abs(results["errors"]) <= 2)*100:.2f}%\n')

    return results

if __name__ == '__main__':
    base_dir = create_results_directories()
    
    # Test both methods for both SNR cases
    for snr_case in ['high', 'low']:
        test_file = f'data/test_data_{snr_case}.h5'
        
        print(f"\nTesting with original interpolation method ({snr_case} SNR):")
        results_interp = test_model(test_file, snr_case)
        
        print(f"\nTesting with cyclic shift method ({snr_case} SNR):")
        results_cycshift = test_model_cycshift_window(test_file, snr_case)
        
        # Generate comparative analysis
        analyze_performance_comparison(results_interp, results_cycshift, snr_case, base_dir)
        
        print(f"\nResults have been saved in the following directories:")
        print(f"- Interpolation method: test_results/interpolation")
        print(f"- Cyclic shift method: test_results/cyclic_shift")
        print(f"- Comparative analysis: test_results/comparative_analysis")