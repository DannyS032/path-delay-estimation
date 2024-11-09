import torch
import h5py
from torch.utils.data import Dataset
import numpy as np

class ChannelDataset(Dataset):

    def __init__(self, h5_file_path):
        with h5py.File(h5_file_path, 'r') as f:
            # Get data from file
            self.cir_l = np.array(f['CIR_Low'])
            self.cir_h = np.array(f['CIR_High'])
            self.cfr_h = np.array(f['CFR_High'])
            self.toa = np.array(f['ToA'])

        # Convert to PyTorch tensors
        self.cir_l = torch.tensor(self.cir_l, dtype=torch.float32)
        self.cir_h = torch.tensor(self.cir_h, dtype=torch.float32)
        self.cfr_h = torch.tensor(self.cfr_h, dtype=torch.float32)
        self.toa = torch.tensor(self.toa, dtype=torch.float32)

    def __len__(self):
        # Return length i.e. number of samples
        return len(self.cir_h)
    
    def __getitem__(self, index):
        # Retrieve channel data for the given index
        cir_l = self.cir_l[index]
        cir_h = self.cir_h[index]
        cfr_h = self.cfr_h[index]
        toa = self.toa[index]

        sample = {'cir_l': cir_l, 'cir_h': cir_h,'cfr_h': cfr_h, 'toa': toa}

        return sample

if __name__ == '__main__':
    file = h5py.File('train_data_gen_high.h5', 'r')
    print(file.keys())