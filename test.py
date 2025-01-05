import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader, Subset
import os
import matplotlib.pyplot as plt
from torchinterp1d import Interp1d
from data.dataset import *
from model.network import *
from test_gen import *
from datetime import datetime

def interp1d(y, x, x_new):
    '''
    linear interpolation for the third dimension
    y: Nx2xL
    x: Nx2xL
    x_new: Nx2xL
    '''
    N = y.shape[0]
    out = []
    for i in range(N):
        y_new = Interp1d.apply(x[i], y[i], x_new[i]) 
        out.append(y_new.unsqueeze(0))
            
    return torch.cat(out, 0).detach()

def test_toa_estimation(test_file, folder_reg, folder_gen, num_test_samples=None, num_plots=10, batch_size=100, win_siz=100, win_idx=128, up_sample=2):
    """
    Testing function for the trained ToA estimation models.
    
    Args:
        test_file: Path to the test dataset
        folder_reg: Directory containing trained models
        num_test_samples: Number of samples to test (None for all samples)
        num_plots: Number of samples to plot and save
        batch_size: Batch size for testing
        win_siz: Window size for cropping the CIR
        win_idx: Number of sample points to input for regressor B
        up_sample: Up sampling factor of the generative network
    """
    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    full_dataset = ChannelDataset(test_file)
    
    # Control number of test samples
    if num_test_samples is not None:
        num_test_samples = min(num_test_samples, len(full_dataset))
        indices = np.random.choice(len(full_dataset), num_test_samples, replace=False)
        dataset = Subset(full_dataset, indices)
    else:
        dataset = full_dataset
        num_test_samples = len(dataset)
    
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    # Initialize models
    signal_length = 128 * up_sample
    gen_model = unet().to(device)
    regA_model = regnet(signal_length).to(device)
    regB_model = regnet(win_idx).to(device)
    
    # Determine SNR case from filename
    if 'high' in test_file:
        snr_case = 'high'
    elif 'low' in test_file:
        snr_case = 'low'
    else:
        snr_case = ''
    
    # Create timestamp for unique saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join('test_results', f'test_run_{timestamp}_{snr_case}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load trained models
    gen_model.load_state_dict(torch.load(os.path.join(folder_gen, f'gen_model_trained_{snr_case}.w'), weights_only=True))
    regA_model.load_state_dict(torch.load(os.path.join(folder_reg, f'regA_model_trained_{snr_case}.w'), weights_only=True))
    regB_model.load_state_dict(torch.load(os.path.join(folder_reg, f'regB_model_trained_{snr_case}.w'), weights_only=True))
    
    # Set models to evaluation mode
    gen_model.eval()
    regA_model.eval()
    regB_model.eval()
    
    # Arrays to store results
    toa_gt = []
    toa_coarse = []
    toa_fine = []
    mse_coarse = []
    mse_fine = []
    
    # Counter for plots
    plots_generated = 0
    
    with torch.no_grad():
        for i, frame in enumerate(test_loader):
            # Move batch to device
            cir_l = frame['cir_l'].float().to(device)
            cir_h = frame['cir_h'].float().to(device)
            toa = frame['toa'].float().to(device)
            
            # Generate High noiseless CIR
            cir_h_pred = gen_model(cir_l)
            
            # Coarse ToA estimation
            toa_coarse_pred = regA_model(cir_h_pred)
            toa_coarse_pred = toa_coarse_pred.squeeze(-1)
            
            # Fine ToA estimation
            index = ((toa_coarse_pred > win_siz//2) * (toa_coarse_pred < 3200 - win_siz//2)).squeeze()
            x = 12.5 * torch.arange(0, cir_h.shape[2]).repeat(2, 1).to(device)
            window = torch.arange(-win_siz, win_siz, 2*win_siz/win_idx).repeat(2, 1).to(device)
            
            if torch.sum(index) != 0:
                cir_h_pred_val = cir_h_pred[index]
                toa_coarse_val = toa_coarse_pred[index]
                x_new = toa_coarse_val.unsqueeze(-1) + window.unsqueeze(0)
                x = x.repeat(cir_h_pred_val.shape[0], 1, 1)
                cir_h_crop = interp1d(cir_h_pred_val, x, x_new)
                
                # Fine adjustment
                regB_pred = regB_model(cir_h_crop).squeeze(-1)
                toa_fine_val = toa_coarse_val + regB_pred
                
                # Store results
                toa_gt.extend(toa[index].cpu().numpy())
                toa_coarse.extend(toa_coarse_val.cpu().numpy())
                toa_fine.extend(toa_fine_val.cpu().numpy())
                
                # Calculate MSE for this batch
                mse_coarse.append(((toa[index] - toa_coarse_val) ** 2).mean().item())
                mse_fine.append(((toa[index] - toa_fine_val) ** 2).mean().item())
                
                # Generate plots if needed
                if plots_generated < num_plots:
                    for j in range(min(num_plots - plots_generated, len(index))):
                        plt.figure(figsize=(16, 8))
                        
                        # Plot original CIR
                        plt.subplot(1, 3, 1)
                        cir_l_plot = real2complex2dim(cir_l[j]).cpu().numpy().squeeze()
                        cir_h_plot = real2complex2dim(cir_h[j]).cpu().numpy().squeeze()
                        x_plot = 12.5 * np.arange(0, 256)
                        plt.plot(x_plot[:70], np.abs(cir_l_plot[:70]), label="Low-res. CIR")
                        plt.plot(x_plot[:70], np.abs(cir_h_plot[:70]), label="Ground Truth High-res. CIR")
                        plt.axvline(toa[j].item(), color='black', linestyle='--', label=f'Ground Truth ToA')
                        plt.axvline(toa_coarse_pred[j].item(), color='red', linestyle='--', label=f'Coarse ToA')
                        if index[j]:
                            plt.axvline(toa_fine_val[torch.where(index)[0] == j].item(), 
                                      color='green', linestyle='--', label=f'Fine ToA')
                        plt.title("CIR Comparison")
                        plt.legend()
                        
                        # Plot generated CIR
                        plt.subplot(1, 3, 2)
                        cir_h_pred_plot = real2complex2dim(cir_h_pred[j]).cpu().numpy().squeeze()
                        plt.plot(x_plot[:70], np.abs(cir_h_pred_plot[:70]), label="Generated High-res. CIR")
                        plt.plot(x_plot[:70], np.abs(cir_h_plot[:70]), label="Ground Truth High-res. CIR")
                        plt.axvline(toa[j].item(), color='black', linestyle='--', label=f'Ground Truth ToA')
                        plt.title("Generated vs Ground Truth CIR")
                        plt.legend()
                        
                        # Plot cropped CIR
                        if index[j]:
                            plt.subplot(1, 3, 3)
                            idx = torch.where(index)[0] == j
                            cir_h_crop_plot = real2complex2dim(cir_h_crop[idx]).cpu().numpy().squeeze()
                            x_new_plot = x_new[idx].cpu().numpy().squeeze()[0,:]
                            plt.plot(x_new_plot, np.abs(cir_h_crop_plot), label="Cropped CIR")
                            plt.axvline(toa[j].item(), color='black', linestyle='--', label=f'Ground Truth ToA')
                            plt.axvline(toa_coarse_val[idx].item(), color='red', linestyle='--', label=f'Coarse ToA')
                            plt.axvline(toa_fine_val[idx].item(), color='green', linestyle='--', label=f'Fine ToA')
                            plt.title("Cropped CIR for Fine Estimation")
                            plt.legend()
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(save_dir, f'sample_{plots_generated + 1}.png'))
                        plt.close()
                        plots_generated += 1
    
    # Convert results to numpy arrays
    toa_gt = np.array(toa_gt)
    toa_coarse = np.array(toa_coarse)
    toa_fine = np.array(toa_fine)
    
    # Calculate overall metrics
    overall_mse_coarse = np.mean(mse_coarse)
    overall_mse_fine = np.mean(mse_fine)
    rmse_coarse = np.sqrt(overall_mse_coarse)
    rmse_fine = np.sqrt(overall_mse_fine)
    
    # Print results
    print(f"\nResults for {snr_case} SNR case:")
    print(f"Number of samples tested: {num_test_samples}")
    print(f"Coarse Estimation RMSE: {rmse_coarse:.4f}")
    print(f"Fine Estimation RMSE: {rmse_fine:.4f}")
    
    # Plot error distributions (absolute)
    plt.figure(figsize=(10, 6))
    plt.hist(np.abs(toa_gt - toa_coarse), bins=50, alpha=0.5, label='Coarse Estimation Error')
    plt.hist(np.abs(toa_gt - toa_fine), bins=50, alpha=0.5, label='Fine Estimation Error')
    plt.xlabel('Absolute Error')
    plt.ylabel('Count')
    plt.title(f'ToA Estimation Absolute Error Distribution ({snr_case} SNR)')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'absolute_error_distribution.png'))
    plt.close()
    
    # Plot error distributions (signed)
    plt.figure(figsize=(10, 6))
    plt.hist(toa_gt - toa_coarse, bins=50, alpha=0.5, label='Coarse Estimation Error')
    plt.hist(toa_gt - toa_fine, bins=50, alpha=0.5, label='Fine Estimation Error')
    plt.xlabel('Signed Error (Ground Truth - Estimated)')
    plt.ylabel('Count')
    plt.title(f'ToA Estimation Signed Error Distribution ({snr_case} SNR)\nNegative = Overestimation, Positive = Underestimation')
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'signed_error_distribution.png'))
    plt.close()
    
    return rmse_coarse, rmse_fine

if __name__ == '__main__':
    # Test parameters
    NUM_TEST_SAMPLES = 5000  # Set to None to test all samples
    NUM_PLOTS = 20          # Number of sample visualizations to save
    
    # Test both SNR cases
    test_file_high = 'data/test_data_high.h5'
    test_file_low = 'data/test_data_low.h5'
    
    rmse_coarse_high, rmse_fine_high = test_toa_estimation(
        test_file_high, 
        'train-reg-1m',
        'train-gen-1m',
        num_test_samples=NUM_TEST_SAMPLES,
        num_plots=NUM_PLOTS
    )
    
    rmse_coarse_low, rmse_fine_low = test_toa_estimation(
        test_file_low, 
        'train-reg-1m',
        'train-gen-1m',
        num_test_samples=NUM_TEST_SAMPLES,
        num_plots=NUM_PLOTS
    )