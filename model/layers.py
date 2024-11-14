import numpy as np 
import torch
import torch.nn as nn

# For Regression Networks
class Flatten(nn.Module):
  def forward(self, x):
    N, C, H = x.size() # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

# For Regression Networks
class Sandwich(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride=1):
        super(Sandwich, self).__init__()
        self.conv = nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.batch = nn.BatchNorm1d(C_out)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.relu(self.batch(self.conv(x)))

# For Generative Network
class SubPixel1D(nn.Module):
    ''' One dimensional subpixel upsampling layer

        Input: N x rC x L =>  N x C x rL
    '''
    def __init__(self, r):
        super(SubPixel1D, self).__init__()
        self.up_factor = r

    def forward(self, x):
        N, rC, L = x.size()
        assert rC % self.up_factor == 0
        C = rC//self.up_factor
        x = x.view(N, C, self.up_factor, L)
        x = x.permute(0,1,3,2)
        x = x.contiguous().view(N, C, -1)
        return x

      
if __name__ == "__main__":

    SubPixel = SubPixel1D(3)
    x1 = torch.ones(40, 1, 30)
    x2 = 2*torch.ones(40, 1, 30)
    x3 = 3*torch.ones(40, 1, 30)
    x = torch.cat((x1,x2,x3), 1)
    
    y = SubPixel(x)