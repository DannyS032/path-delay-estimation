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
from train_gen import *

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

def train_regs_test_network(regA, regB, gen, train_loader, num_epochs, folder, snr_case, learning_rate=1e-3, device='cuda', step=40, print_interval=100, win_siz = 100, win_idx = 128, test_plots=False):
    """
    Training function for the Regressive A-B Cascade network.
    
    Args:
        regA: Regressor A model of the Regressive network model
        regB: Regressor B model of the Regressive network model
        gen: U-net Generative network model (trained)
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
        models: Trained Regressors A & B of  Regressive network
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
                cir_h = frame['cir_h'].float().to(device)
                toa = frame['toa'].float().to(device)

                # Reset gradient
                optimizer.zero_grad()

                # Generate High noiseless CIR through Generative network
                cir_h_pred = gen(cir_l)

                # Coarse ToA estimation
                toa_coarse = regA(cir_h_pred)

                # Cropping CIR for Regressor B
                index = ((toa_coarse > win_siz//2) * (toa_coarse < 3200 - win_siz//2)).squeeze()
                x = 12.5 * torch.arange(0, cir_h.shape[2]).repeat(2, 1).to(device)
                window = torch.arange(-win_siz, win_siz, 2*win_siz/win_idx).repeat(2, 1).to(device)

                if torch.sum(index) != 0:
                    # For valid samples with coarse ToA estimation in range cont. to fine estimation
                    with torch.no_grad():
                        cir_h_val = cir_h[index]
                        cir_h_pred_val = cir_h_pred[index]
                        toa_coarse_val = toa_coarse[index]
                        x_new = toa_coarse_val.unsqueeze(-1) + window.unsqueeze(0)
                        x = x.repeat(cir_h_pred_val.shape[0],1,1)
                        cir_h_crop = interp1d(cir_h_pred_val, x, x_new) 


                    # Estimating error of coarse ToA from ground truth
                    toa_val = toa[index]
                    toa_coarse_val = toa_coarse_val.squeeze(-1)
                    diff_gt_coarse = toa_val - toa_coarse_val # Difference between ground-truth tau_0 and coarse[tau_0] 
                    regB_pred = regB(cir_h_crop)
                    regB_pred = regB_pred.squeeze(-1)
                    toa_fine = toa_coarse_val + regB_pred

                    rand_index = random.randint(0, cir_h_crop.size(0)-1) # [0, N-1]
                    if test_plots and (i % 400) == 0:
                        # Test plots
                        cir_l_plot = real2complex2dim(cir_l[rand_index]).cpu().numpy().squeeze()
                        cir_h_plot = real2complex2dim(cir_h[rand_index]).cpu().numpy().squeeze()
                        cir_h_pred_plot = real2complex2dim(cir_h_pred[rand_index]).detach().cpu().numpy().squeeze()
                        cir_h_crop_plot = real2complex2dim(cir_h_crop[rand_index]).detach().cpu().numpy().squeeze()
                        x_plot = 12.5 * torch.arange(0, 256)
                        x_new_plot = x_new[rand_index].cpu().numpy().squeeze()
                        x_new_plot = x_new_plot[0,:]
                        toa_plot = toa[rand_index].item()
                        toa_coarse_plot = toa_coarse_val[rand_index].item()
                        toa_fine_plot = toa_fine[rand_index].item()

                        # abs values
                        cir_l_plot = np.abs(cir_l_plot)
                        cir_h_plot = np.abs(cir_h_plot)
                        cir_h_pred_plot = np.abs(cir_h_pred_plot)
                        cir_h_crop_plot = np.abs(cir_h_crop_plot)

                        plt.figure(figsize=(16,8))
                        # Input Plot
                        plt.subplot(1, 3, 1)
                        plt.plot(x_plot[:70], cir_l_plot[:70], label="Low-res. CIR")
                        plt.plot(x_plot[:70], cir_h_plot[:70], label="Ground Truth High-res. CIR")
                        plt.axvline(toa_plot, linewidth = 2, linestyle ="--", color ='black',label=f'ToA {toa_plot}')
                        plt.axvline(toa_coarse_plot, linewidth = 2, linestyle ="--", color ='red',label=f'Coarse ToA {toa_coarse_plot}')
                        plt.axvline(toa_fine_plot, linewidth = 2, linestyle ="--", color ='green',label=f'Fine ToA {toa_fine_plot}')                        
                        plt.title(f"Noisy low-res. CIR")
                        plt.legend()
                        
                        # Target Plot
                        plt.subplot(1, 3, 2)
                        plt.plot(x_plot[:70], cir_h_pred_plot[:70], label="Generated High-res. CIR")
                        plt.plot(x_plot[:70], cir_h_plot[:70], label="Ground Truth High-res. CIR")
                        plt.axvline(toa_plot, linewidth = 2, linestyle ="--", color ='black',label=f'ToA {toa_plot}')
                        plt.axvline(toa_coarse_plot, linewidth = 2, linestyle ="--", color ='red',label=f'Coarse ToA {toa_coarse_plot}') 
                        plt.axvline(toa_fine_plot, linewidth = 2, linestyle ="--", color ='green',label=f'Fine ToA {toa_fine_plot}')                                               
                        plt.title(f"Generated noiseless high-res. CIR")
                        plt.legend()

                        # Prediction Plot with Target Overlay
                        plt.subplot(1, 3, 3)
                        plt.plot(x_new_plot, cir_h_crop_plot, label="Cropped CIR (High)", color='g')
                        plt.axvline(toa_plot, linewidth = 2, linestyle ="--", color ='black',label=f'ToA {toa_plot}')
                        plt.axvline(toa_coarse_plot, linewidth = 2, linestyle ="--", color ='red',label=f'Coarse ToA {toa_coarse_plot}')                        
                        plt.axvline(toa_fine_plot, linewidth = 2, linestyle ="--", color ='green',label=f'Fine ToA {toa_fine_plot}')                        
                        plt.title(f"Cropped (windowed) high-res. CIR around coarse ToA")
                        plt.legend()

                        # Show the plots
                        plt.tight_layout()
                        plt.savefig(f'train-reg-1m/CIR_ToA_plots_{snr_case}.png')
                        plt.close()
                    
                    # Compute loss (fine + coarse)
                    toa_coarse = toa_coarse.squeeze(-1)
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
                    toa_coarse = toa_coarse.squeeze(-1)
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

def train(training_file, folder_reg, folder_gen, batch_size=400, num_epochs=200, win_idx=128, up_sample=2, plot_loss=False):
    # choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset with dataloader
    proj_directory = os.getcwd()
    file_path = os.path.join(proj_directory, training_file)
    dataset = ChannelDataset(file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    # initialize models
    signal_length = 128 * up_sample
    gen_model = unet().to(device)
    regA_model = regnet(signal_length).to(device)
    regB_model = regnet(win_idx).to(device)

    # initialize folder for saving training data
    folder_reg = os.path.join(proj_directory, folder_reg)
    if os.path.exists(folder_reg) == False:
        os.makedirs(folder_reg)
    
    # Checking SNR case
    if 'high' in training_file:
        snr_case = 'high'
    elif 'low' in training_file:
        snr_case = 'low'
    else:
        snr_case = ''
    
    # train generative model
    gen_model_trained, loss_reg = train_gen_network(gen_model, dataloader, num_epochs, folder_gen, snr_case)
    gen_model_trained.eval()

    # train regressors
    loss_reg, loss_A, loss_B = train_regs_test_network(regA_model, regB_model, gen_model_trained, dataloader, num_epochs, folder_reg, snr_case, test_plots=True)

    if plot_loss:
        log_loss_reg = np.log(loss_reg)
        log_loss_A = np.log(loss_A)
        log_loss_B = np.log(loss_B)
        plt.figure(figsize=(8, 8))
        plt.plot(log_loss_reg, label="Total loss (corase+fine)")
        plt.plot(log_loss_A, label="Coarse loss")
        plt.plot(log_loss_B, label="Fine loss")
        plt.xlabel("Iteration")
        plt.ylabel("Log(Loss)")
        plt.title("Log Loss vs. Iteration (Regressors Network)")
        plt.legend()
        plt.grid()
        plt.savefig(f'train-reg-1m/loss_vs_interation_{snr_case}_plot.png')
        plt.close()

if __name__ == '__main__':
    
    file_name_high = 'data/train_data_high_1m.h5'
    file_name_low = 'data/train_data_low_1m.h5'

    train(file_name_high, folder_reg='train-reg-1m', folder_gen='train-gen-1m')
    train(file_name_low, folder_reg='train-reg-1m', folder_gen='train-gen-1m')