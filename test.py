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
import random
from tqdm import tqdm

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

def plot_sample(cir_l, cir_h, cir_h_pred, cir_h_crop, x_new, toa, toa_coarse, toa_fine, snr_case, sample_idx):
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
    os.makedirs('test_results', exist_ok=True)
    plt.savefig(f'test_results/CIR_ToA_plots_{snr_case}_sample_{sample_idx}.png')
    plt.close()

def test_model(test_file, snr_case, num_plots=20):
    """Run inference testing on the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    win_siz = 100
    win_idx = 128
    
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

    # Arrays to store results
    toa_errors = []
    plot_indices = random.sample(range(len(dataset)), num_plots)

    with torch.no_grad():
        for i, frame in enumerate(tqdm(dataloader, desc="Testing")):
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

            # Calculate error
            error = toa_fine - toa
            toa_errors.append(error.item())

            # Generate plots for selected samples
            if i in plot_indices:
                plot_sample(cir_l[0], cir_h[0], cir_h_pred[0], cir_h_crop[0], 
                          x_new[0], toa[0].item(), toa_coarse[0].item(), 
                          toa_fine[0].item(), snr_case, i)

    # Convert errors to numpy array
    toa_errors = np.array(toa_errors)

    # Calculate RMSE
    rmse = np.sqrt(np.mean(toa_errors**2))

    # Plot error histogram
    plt.figure(figsize=(10, 6))
    plt.hist(toa_errors, bins=50, density=True, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    plt.xlabel('ToA Estimation Error (True - Estimated)')
    plt.ylabel('Density')
    plt.title(f'ToA Estimation Error Distribution ({snr_case} SNR)\nRMSE: {rmse:.2f}')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'test_results/error_distribution_{snr_case}.png')
    plt.close()

    # Save numerical results
    with open(f'test_results/metrics_{snr_case}.txt', 'w') as f:
        f.write(f'RMSE: {rmse:.4f}\n')
        f.write(f'Mean Error: {np.mean(toa_errors):.4f}\n')
        f.write(f'Median Error: {np.median(toa_errors):.4f}\n')
        f.write(f'Std Dev: {np.std(toa_errors):.4f}\n')
        f.write(f'Max Error: {np.max(np.abs(toa_errors)):.4f}\n')
        f.write(f'Samples within ±1: {np.mean(np.abs(toa_errors) <= 1)*100:.2f}%\n')
        f.write(f'Samples within ±2: {np.mean(np.abs(toa_errors) <= 2)*100:.2f}%\n')

if __name__ == '__main__':
    # Test both SNR cases
    for snr_case in ['high', 'low']:
        test_file = f'data/test_data_{snr_case}.h5'
        test_model(test_file, snr_case)