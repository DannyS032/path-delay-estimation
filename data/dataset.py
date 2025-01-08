import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class ChannelDataset(Dataset):

    def __init__(self, h5_file_path):
        with h5py.File(h5_file_path, 'r') as f:
            # Get data from file
            self.cir_l = np.transpose(f['CIR_Low'], (2, 1, 0))
            self.cir_h = np.transpose(f['CIR_High'], (2, 1, 0))
            self.cfr_h = np.transpose(f['CFR_High'], (2, 1, 0))
            self.toa = np.transpose(f['ToA'])
            if 'L' in f:
                self.l = np.transpose(f['L'])
            else:
                self.l = None
            if 'SNR' in f:
                self.snr = np.transpose(f['SNR'])
            else:
                self.snr = None

    def __len__(self):
        # Return length i.e. number of samples
        return self.toa.shape[0]
    
    def __getitem__(self, index):
        # Retrieve channel data for the given index
        cir_l = self.cir_l[index]
        cir_h = self.cir_h[index]
        cfr_h = self.cfr_h[index]
        toa = self.toa[index].squeeze()

        if hasattr(self, 'l') and hasattr(self, 'snr'):
            l = self.l[index].squeeze()
            snr = self.snr[index].squeeze()

            sample = {'cir_l': cir_l, 'cir_h': cir_h,'cfr_h': cfr_h, 'toa': toa, 'l': l, 'snr': snr}
        else:
            sample = {'cir_l': cir_l, 'cir_h': cir_h,'cfr_h': cfr_h, 'toa': toa}

        return sample

if __name__ == '__main__':

    proj_directory = os.getcwd()
    file_path = os.path.join(proj_directory, 'data/test_data_high.h5')
    dataset = ChannelDataset(file_path)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    for i, batch in enumerate(dataloader):
        # Print out the shape of the batch
        print(f"Batch {i + 1}:")
        print(f"cir_l shape: {batch['cir_l'].shape}")
        print(f"cir_h shape: {batch['cir_h'].shape}")
        print(f"cfr_h shape: {batch['cfr_h'].shape}")
        print(f"toa shape: {batch['toa'].shape}")
        print(f"ToA: {batch['toa'][0]}")
        print(f"L: {batch['l'][0]}")
        print(f"SNR: {batch['snr'][0]}")

        if i==5:
            break
    