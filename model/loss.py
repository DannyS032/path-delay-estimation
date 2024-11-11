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
      pred_complex = real2complex(prediction) # (N, 1, L) complex
      pred_cfr = torch.fft.fft(pred_complex, dim=-1, norm='ortho')
      targ_complex = real2complex(target) # (N, 1, L) complex
      targ_cfr = targ_complex

      diff = pred_cfr - targ_cfr
      squared_norm = torch.abs(diff)**2
      loss = torch.mean(squared_norm)

      return loss

def amploss(prediction, target):
      """
    Compute L2 norm squared loss of amplitude in time 
    
    Args:
        prediction: Real tensor of shape (N, 2, L) 
        target: Real tensor of shape (N, 2, L)
    
    Returns:
        Scalar loss value (mean over batch)
    """
      pred_cir = real2complex(prediction) # (N, 1, L) complex
      targ_cir = real2complex(target) # (N, 1, L) complex

      pred_amp_sq = torch.abs(pred_cir)**2
      targ_amp_sq = torch.abs(targ_cir)**2

      amp_diff = pred_amp_sq - targ_amp_sq

      loss = torch.mean(amp_diff**2)

      return loss

def combloss(prediction_cir, target_cir, target_cfr, alpha = 0.5):
       """
    Combined losses of amplitude and frequence with weight factor alpha 
    
    Args:
        prediction_cir: Real CIR prediction tensor of shape (N, 2, L) 
        target_cir: Real CIR target tensor of shape (N, 2, L)
        target_cfr: Real CFR target tensor of shape (N, 2, L)
        alpha: Weight for frequency (1-alpha for time)
    Returns:
        Combined scalar loss value
    """
       freq_loss = freqloss(prediction_cir, target_cfr)
       amp_loss = amploss(prediction_cir, target_cir)
       
       loss = alpha * freq_loss + (1-alpha) * amp_loss

       return loss
       


if __name__ == '__main__':
    # Create sample data
    N, L = 10, 256
    pred = torch.randn(N, 2, L)
    target_cir = torch.randn(N, 2, L)
    target_cfr = torch.randn(N, 2, L)
    
    # Compute individual losses
    freq_loss = freqloss(pred, target_cfr)
    amp_loss = amploss(pred, target_cir)
    total_loss = combloss(pred, target_cir, target_cfr, alpha=0.5)
    
    print(f"Complex L2 Loss: {freq_loss.item():.4f}")
    print(f"Amplitude Squared Loss: {amp_loss.item():.4f}")
    print(f"Combined Loss (alpha=0.5): {total_loss.item():.4f}")