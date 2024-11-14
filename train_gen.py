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
        for epoch in range(num_epochs):
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
                    print(log_message)
                    log_file.write(log_message + '\n')
                    log_file.flush()

                loss_array.append(loss.item())
            
            exp_lr_scheduler.step()

    print('Training Generative network finished!')
    torch.save(model.state_dict(), model_path)

    return model, loss_array

def train(training_file, batch_size=400, num_epochs=200, plot_losses=False, alpha=0.5):
    # choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset with dataloader
    proj_directory = os.getcwd()
    gen_file_path = os.path.join(proj_directory, training_file)
    gen_dataset = ChannelDataset(gen_file_path)
    gen_dataloader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    # initialize model
    model = unet().to(device)

    # initialize folder for saving training data
    folder = os.path.join(proj_directory, 'train-gen')
    if os.path.exists(folder) == False:
        os.makedirs(folder)
    
    # Checking SNR case
    if 'high' in training_file:
        snr_case = 'high'
    elif 'low' in training_file:
        snr_case = 'low'
    else:
        snr_case = ''
    
    # train
    model_trained, losses = train_gen_network(model, gen_dataloader, num_epochs, folder, snr_case, alpha=alpha)
    model_trained.eval()

    # Plots for testing
    log_loss = np.log(losses)
    plt.figure
    plt.plot(log_loss, label="loss")
    plt.xlabel("Iteration")
    plt.ylabel("Log(Loss)")
    plt.title("Log Loss vs. Iteration")
    plt.legend()
    plt.grid()
    if plot_losses:
        plt.savefig(f'train-gen/loss_vs_interation_{snr_case}_plot.png')
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

    train(file_name_high)
    train(file_name_low)

   



