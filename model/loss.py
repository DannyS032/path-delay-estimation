import torch

def real2complex(x):
     """
    Convert real tensor of shape (N, 2, L) to complex tensor of shape (N, 1, L)
    where channel 0 is real part and channel 1 is imaginary part
    """
     return torch.complex(x[:, 0:1, :], x[:, 1:2, :])

def freqloss(prediction, target):
      """
    Compute L2 norm squared loss in frequency
    
    Args:
        prediction: Real tensor of shape (N, 2, L) 
        target: Real tensor of shape (N, 2, L)
    
    Returns:
        Scalar loss value (mean over batch)
    """
      pred_complex = real2complex(prediction)
      pred_cfr = torch.fft.fft(pred_complex, dim=-1, norm='ortho')
      targ_complex = real2complex(target)
      targ_cfr = targ_complex

      diff = pred_cfr - targ_cfr
      squared_norm = torch.abs(diff)**2
      loss = torch.mean(torch.sum(squared_norm, dim=2))

      return loss


if __name__ == '__main__':
     tensor = torch.rand(10, 2, 256)
     tensor2 = torch.rand(10, 2, 256)
     loss = freqloss(tensor, tensor2)
     print(loss)