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

def train_gen_network(model, train_loader, num_epochs, folder, learning_rate=1e-3, alpha=0.5, device='cuda', step=40, print_interval=100):
    """
    Training function for the U-net Generative network.
    
    Args:
        model: The U-net Generative network model
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate for optimizer
        alpha: Weight factor for combined loss
        device: Device to run the training on
        step: Step size for learning rate decay
        print_interval: Interval for logging training status
        folder: Directory path for saving training logs and the model
    Returns:
        Trained U-net Generative network model
    """
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.3)
    
    # Files for saving loss values & trained model
    log_path = os.path.join(folder, 'loss_log_gen.txt')
    model_path = os.path.join(folder, 'model_gen_trained.w')
    
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
            
            exp_lr_scheduler.step()

    print('Training Generative network finished!')
    torch.save(model.state_dict(), model_path)

    return model

def train(training_file, batch_size=400, num_epochs=200):
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
    
    # train
    model_trained = train_gen_network(model, gen_dataloader, num_epochs, folder)
    model_trained.eval()


if __name__ == '__main__':
    
    file_name = 'data/train_data_gen_high.h5'
    train(file_name)
