import torch
import numpy as np
import h5py
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import *
from model.layers import *
from model.network import *
from model.loss import *
import os
import matplotlib.pyplot as plt
import random

def plot2test(cir_l, cir_h_gt, cir_h_pred):

    # abs values
    cir_l = np.abs(cir_l)
    cir_h_gt = np.abs(cir_h_gt)
    cir_h_pred = np.abs(cir_h_pred)

    plt.figure(figsize=(5,5))
    plt.plot(cir_h_gt[:50], label="Ground-truth CIR (High)", color='g')
    plt.plot(cir_h_pred[:50], label="Generated CIR (High)", color='r')
    plt.title("Ground-truth vs Generated CIR")
    plt.legend()

    plt.figure(figsize=(12,4))
    # Input Plot
    plt.subplot(1, 3, 1)
    plt.plot(cir_l[:70], label="Low-res. CIR")
    plt.title("Noisy low-res. CIR")
    plt.legend()
    
    # Target Plot
    plt.subplot(1, 3, 2)
    plt.plot(cir_h_gt[:70], label="High-res. CIR", color='g')
    plt.title("Ground-truth high-res. CIR")
    plt.legend()
    
    # Prediction Plot with Target Overlay
    plt.subplot(1, 3, 3)
    plt.plot(cir_h_gt[:70], label="Ground-truth CIR (High)", color='g')
    plt.plot(cir_h_pred[:70], label="Generated CIR (High)", color='r')
    plt.title("Ground-truth vs Generated CIR")
    plt.legend()
    
    # Show the plots
    plt.tight_layout()
    plt.show()

def real2complex2dim(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a real tensor of shape (2, L) to a complex tensor of shape (1, L)
    where channel 0 contains real parts and channel 1 contains imaginary parts.
    
    Args:
        x (torch.Tensor): Input tensor of shape (2, L)
        
    Returns:
        torch.Tensor: Complex tensor of shape (1, L)
    """
    assert x.shape[0] == 2, f"Expected first dimension to be 2, got {x.shape[0]}"
    
    # Extract real and imaginary parts
    real_part = x[0]  # Shape: (L,)
    imag_part = x[1]  # Shape: (L,)
    
    # Create complex tensor
    complex_tensor = torch.complex(real_part, imag_part)  # Shape: (L,)
    
    # Add batch dimension
    return complex_tensor.unsqueeze(0)  # Shape: (1, L)

def test_rand_sample(test_file, snr_case, batch_size=400):

    # choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset with dataloader
    proj_directory = os.getcwd()
    gen_test_file_path = os.path.join(proj_directory, test_file)
    gen_test_dataset = ChannelDataset(gen_test_file_path)
    gen_test_dataloader = DataLoader(gen_test_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    # load trained model
    model = unet().to(device)
    gen_model_path = os.path.join(proj_directory, f'train-gen/gen_model_trained_{snr_case}.w')
    model.load_state_dict(torch.load(gen_model_path, weights_only=True))
    model.eval()

    # input data through model and get predictions
    for batch in gen_test_dataloader:
        # Move batch to device
        cir_l_batch = batch['cir_l'].float().to(device)  # Shape: (N, 2, L)
        cir_h_batch = batch['cir_h'].float().to(device)  # Shape: (N, 2, L)

        with torch.no_grad():
            predicted_cir_h_batch = model(cir_l_batch)  # Shape: (N, 2, L)
        
        # select random sample from batch
        rand_index = random.randint(0, cir_l_batch.size(0)-1) # [0, N-1]
        cir_l_real = cir_l_batch[rand_index]
        cir_h_real = cir_h_batch[rand_index]
        predicted_cir_h_real = predicted_cir_h_batch[rand_index]

        # convert to complex tensors
        cir_l_comp = real2complex2dim(cir_l_real)
        cir_h_comp = real2complex2dim(cir_h_real)
        predicted_cir_h_comp = real2complex2dim(predicted_cir_h_real)

        # convert to numpy arrays
        cir_l = cir_l_comp.cpu().numpy().squeeze()
        cir_h_gndtru = cir_h_comp.cpu().numpy().squeeze()
        cir_h_genertd = predicted_cir_h_comp.cpu().numpy().squeeze()

        plot2test(cir_l, cir_h_gndtru, cir_h_genertd)
        break

if __name__=='__main__':
    test_file_high = 'data/test_data_high.h5'
    test_file_low = 'data/test_data_low.h5'

    test_rand_sample(test_file_high,snr_case='high', batch_size=5000)
    test_rand_sample(test_file_low,snr_case='low', batch_size=5000)

