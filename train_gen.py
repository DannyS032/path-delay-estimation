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
from tqdm import tqdm
from torchinterp1d import Interp1d
import random
from test_gen import *

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

def train_gen_network(model, train_loader, num_epochs, folder, snr_case, learning_rate=1e-3, alpha=0.5, device='cuda', step=40, print_interval=100):
    """
    Training function for the U-net Generative network.
    
    Args:
        model: The U-net Generative network model
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        folder: Directory path for saving training logs and the model
        snr_case: SNR case (either high or low)
        learning_rate: Initial learning rate for optimizer
        alpha: Weight factor for combined loss
        device: Device to run the training on
        step: Step size for learning rate decay
        print_interval: Interval for logging training status
    Returns:
        model: Trained U-net Generative network model
    """
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.3)
    
    # Files for saving loss values & trained model
    log_path = os.path.join(folder, f'gen_loss_log_{snr_case}.txt')
    model_path = os.path.join(folder, f'gen_model_trained_{snr_case}.w')

    # Loss array for later plotting
    loss_array = []
    
    with open(log_path, 'w') as log_file:
        for epoch in tqdm(range(num_epochs)):
            # Training loop
            
            for i, (frame) in enumerate(train_loader):
                # Move batch to device
                cir_l = frame['cir_l'].float().to(device)
                cir_h = frame['cir_h'].float().to(device)
                cfr_h = frame['cfr_h'].float().to(device)
                
                # Forward pass
                predicted_cir_h = model(cir_l)
                
                # Compute loss
                loss = combloss(predicted_cir_h, cir_h, cfr_h, alpha=alpha)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Print progress every print_interval batches
                if (i % print_interval) == 0:
                    log_message = f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}'
                    log_file.write(log_message + '\n')
                    log_file.flush()

                loss_array.append(loss.item())
            
            exp_lr_scheduler.step()

    print('Training Generative network finished!')
    torch.save(model.state_dict(), model_path)

    return model, loss_array

def train_regs_network(regA, regB, gen, train_loader, num_epochs, folder, snr_case, learning_rate=1e-3, device='cuda', step=40, print_interval=100, win_siz = 100, win_idx = 128, test_plots=False):
    """
    Training function for the Regressive A-B Cascade network.
    
    Args:
        regA: Regressor A model of the Regressive network model
        regB: Regressor B model of the Regressive network model
        gen: The Trained U-net Generative network model
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        folder: Directory path for saving training logs and the model
        snr_case: SNR case (either high or low)
        learning_rate: Initial learning rate for optimizer
        device: Device to run the training on
        step: Step size for learning rate decay
        print_interval: Interval for logging training status
        win_siz: Window size for cropping the CIR
        win_idx: Number of sample points to input for regressor B
    Returns:
        model: Trained Regressors A & B of  Regressive network
    """
    
    # Initialize optimizer
    optimizer = optim.Adam(list(regA.parameters())+list(regB.parameters()), lr=learning_rate)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.3)
    
    # Files for saving loss values & trained model
    log_path = os.path.join(folder, f'regA+B_loss_log_{snr_case}.txt')
    log_toa_path = os.path.join(folder, f'regA+B_toa_log_{snr_case}.txt')
    model_path_A = os.path.join(folder, f'regA_model_trained_{snr_case}.w')
    model_path_B = os.path.join(folder, f'regB_model_trained_{snr_case}.w')

    # Loss array for later plotting
    loss_array = []
    loss_arrayA = []
    loss_arrayB = []
    
    with open(log_path, 'w') as log_file, open(log_toa_path, 'w') as log_toa_file:
        for epoch in tqdm(range(num_epochs)):
            # Training loop
            
            for i, (frame) in enumerate(train_loader):
                # Move batch to device
                cir_l = frame['cir_l'].float().to(device)
                toa = frame['toa'].float().to(device)

                # Reset gradient
                optimizer.zero_grad()
                
                # CIR enhancement
                cir_h_gen = gen(cir_l)

                # Coarse ToA estimation
                toa_coarse = regA(cir_h_gen)

                # Cropping CIR for Regressor B
                index = ((toa_coarse > win_siz//2) * (toa_coarse < 3200 - win_siz//2)).squeeze()
                x = 12.5 * torch.arange(0, cir_l.shape[2]).repeat(2, 1).to(device)
                window = torch.arange(-win_siz, win_siz, 2*win_siz/win_idx).repeat(2, 1).to(device)

                if torch.sum(index) != 0:
                    # For valid samples with coarse ToA estimation in range cont. to fine estimation
                    with torch.no_grad():
                        cir_h_gen_val = cir_h_gen[index]
                        toa_coarse_val = toa_coarse[index]
                        x_new = toa_coarse.unsqueeze(-1) + window.unsqueeze(0)
                        x = x.repeat(cir_l.shape[0],1,1)
                        cir_h_gen_crop = interp1d(cir_h_gen_val, x, x_new) 


                    # Estimating error of coarse ToA from ground truth
                    toa_val = toa[index]
                    diff_gt_coarse = toa_val - toa_coarse_val # Difference between ground-truth tau_0 and coarse[tau_0] 
                    regB_pred = regB(cir_h_gen_crop)
                    toa_fine = toa_coarse_val + regB_pred

                    rand_index = random.randint(0, cir_h_gen_crop.size(0)-1) # [0, N-1]
                    if test_plots and (i % 200) == 0 and (epoch % 5) == 0:
                        # Test plots
                        cir_l_plot = real2complex2dim(cir_l[rand_index]).cpu().numpy().squeeze()
                        cir_h_gen_plot = real2complex2dim(cir_h_gen[rand_index]).detach().cpu().numpy().squeeze()
                        cir_h_gen_crop_plot = real2complex2dim(cir_h_gen_crop[rand_index]).detach().cpu().numpy().squeeze()
                        x_plot = 12.5 * torch.arange(0, 256)
                        x_new_plot = x_new[rand_index].cpu().numpy().squeeze()
                        x_new_plot = x_new_plot[0,:]
                        toa_plot = toa[rand_index].item()
                        toa_coarse_plot = toa_coarse[rand_index].item()
    
                        # abs values
                        cir_l_plot = np.abs(cir_l_plot)
                        cir_h_gen_plot = np.abs(cir_h_gen_plot)
                        cir_h_gen_crop_plot = np.abs(cir_h_gen_crop_plot)
    
                        plt.figure(figsize=(12,4))
                        # Input Plot
                        plt.subplot(1, 3, 1)
                        plt.plot(x_plot[:70], cir_l_plot[:70], label="Low-res. CIR")
                        plt.axvline(toa_plot, linewidth = 2, linestyle ="--", color ='red',)
                        plt.title(f"Noisy low-res. CIR, ToA: {toa_plot}")
                        plt.legend()
                        
                        # Target Plot
                        plt.subplot(1, 3, 2)
                        plt.plot(x_plot[:70], cir_h_gen_plot[:70], label="High-res. CIR", color='g')
                        plt.axvline(toa_plot, linewidth = 2, linestyle ="--", color ='red',)
                        plt.title(f"Generated high-res. CIR, ToA: {toa_plot}")
                        plt.legend()
                        
                        # Prediction Plot with Target Overlay
                        plt.subplot(1, 3, 3)
                        plt.plot(x_new_plot, cir_h_gen_crop_plot, label="Ground-truth CIR (High)", color='g')
                        plt.axvline(toa_plot, linewidth = 2, linestyle ="--", color ='red',)
                        plt.title(f"Cropped (windowed) high-res. CIR around coarse ToA {toa_coarse_plot}")
                        plt.legend()
                        
                        # Show the plots
                        plt.tight_layout()
                        plt.show()
                    
                    # Compute loss (fine + coarse)
                    coarse_loss = mseloss(toa_coarse, toa)
                    fine_loss = mseloss(regB_pred, diff_gt_coarse)
                    loss = coarse_loss + fine_loss

                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()

                    # Print progress every print_interval batches
                    if (i % print_interval) == 0:
                        log_toa_messege = f'Ground Truth ToA: {toa[rand_index].item()}, Coarse ToA: {toa_coarse[rand_index].item()}, Fine ToA: {toa_fine[rand_index].item()}'
                        log_message = f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Coarse Loss: {coarse_loss.item():.4f}, Fine Loss: {fine_loss.item():.4f}'
                        log_toa_file.write(log_toa_messege + '\n')
                        log_toa_file.flush()
                        log_file.write(log_message + '\n')
                        log_file.flush()

                    loss_array.append(loss.item())
                    loss_arrayA.append(coarse_loss.item())
                    loss_arrayB.append(fine_loss.item())

                else:
                    
                    # Compute loss (only coarse)
                    loss = mseloss(toa_coarse, toa)

                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()

                
                    # Print progress every print_interval batches
                    if (i % print_interval) == 0:
                        log_message = f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], (Coarse) Loss: {loss.item():.4f}'
                        log_file.write(log_message + '\n')
                        log_file.flush()

                    loss_array.append(loss.item())
            
            exp_lr_scheduler.step()

    print('Training Regressive network finished!')
    torch.save(regA.state_dict(), model_path_A)
    torch.save(regB.state_dict(), model_path_B)

    return loss_array, loss_arrayA, loss_arrayB

def train(training_file, batch_size=400, num_epochs=200, plot_losses=False, alpha=0.5, win_idx = 128, reg_test_mode=False):
    # choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset with dataloader
    proj_directory = os.getcwd()
    file_path = os.path.join(proj_directory, training_file)
    dataset = ChannelDataset(file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    # initialize models
    gen_model = unet().to(device)
    regA_model = regnet(256).to(device)
    regB_model = regnet(win_idx).to(device)

    # initialize folder for saving training data
    folder_gen = os.path.join(proj_directory, 'train-gen')
    folder_reg = os.path.join(proj_directory, 'train-reg')
    if os.path.exists(folder_gen) == False:
        os.makedirs(folder_gen)
    if os.path.exists(folder_reg) == False:
        os.makedirs(folder_reg)
    
    # Checking SNR case
    if 'high' in training_file:
        snr_case = 'high'
    elif 'low' in training_file:
        snr_case = 'low'
    else:
        snr_case = ''
    
    # train
    if reg_test_mode:
        gen_model_path = os.path.join(proj_directory, f'train-gen/gen_model_trained_{snr_case}.w')
        gen_model.load_state_dict(torch.load(gen_model_path, weights_only=True))
        gen_model_trained = gen_model.eval()
    else:
        gen_model_trained, losses = train_gen_network(gen_model, dataloader, num_epochs, folder_gen, snr_case, alpha=alpha)
        gen_model_trained.eval()
    loss_reg, loss_regA, loss_regB = train_regs_network(regA_model, regB_model, gen_model_trained, dataloader, num_epochs, folder_reg, snr_case, win_idx=win_idx, test_plots=True)

    # Plots for testing
    log_loss = np.log(losses)
    plt.figure
    plt.plot(log_loss, label="loss")
    plt.xlabel("Iteration")
    plt.ylabel("Log(Loss)")
    plt.title("Log Loss vs. Iteration (Generative Network)")
    plt.legend()
    plt.grid()
    plt.savefig(f'train-gen/loss_vs_interation_{snr_case}_plot.png')

    log_loss_reg = np.log(loss_reg)
    log_loss_regA = np.log(loss_regA)
    log_loss_regB = np.log(loss_regB)
    plt.figure
    plt.plot(log_loss_reg, label="total loss")
    plt.plot(log_loss_regA, label="reg A loss")
    plt.plot(log_loss_regB, label="reg B loss")
    plt.xlabel("Iteration")
    plt.ylabel("Log(Loss)")
    plt.title("Log Loss vs. Iteration (Regressive Network)")
    plt.legend()
    plt.grid()
    plt.savefig(f'train-reg/loss_vs_interation_{snr_case}_plot.png')
    if plot_losses:
        
        plt.show()

    return log_loss

def train_for_test_alpha(file_name_high, file_name_low):
    # Will be deleted after testing
    alpha_values = [0.1, 0.25, 0.5, 0.75, 0.9]
    results_high, results_low = {}, {}
    for alpha in alpha_values:
        loss_high = train(file_name_high, plot_losses=False, alpha=alpha)
        results_high[alpha] = loss_high
        loss_low = train(file_name_low, plot_losses=False, alpha=alpha)
        results_low[alpha] = loss_low
    
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    for alpha, loss in results_high.items():
        plt.plot(loss, label=f'alpha = {alpha}')
    plt.xlabel("Iteration")
    plt.ylabel("Log(Loss)")
    plt.title("Log Loss vs. Iteration (High SNR)")
    plt.legend()
    plt.grid()

    plt.subplot(1,2,2)
    for alpha, loss in results_low.items():
        plt.plot(loss, label=f'alpha = {alpha}')
    plt.xlabel("Iteration")
    plt.ylabel("Log(Loss)")
    plt.title("Log Loss vs. Iteration (Low SNR)")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == '__main__':
    
    file_name_high = 'data/train_data_high.h5'
    file_name_low = 'data/train_data_low.h5'

    train(file_name_high, reg_test_mode=True)
    train(file_name_low, reg_test_mode=True)

   



