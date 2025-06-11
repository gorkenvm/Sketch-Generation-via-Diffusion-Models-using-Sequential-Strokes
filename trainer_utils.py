import torch
import torch.nn.functional as F

def diffusion_train_step(model, batch, optimizer, device='cuda'):
    """
    batch: {'seq': (B, L, 4), ...}
    """
    x_0 = batch['seq'].to(device)     
    noise = torch.randn_like(x_0)
    t = torch.rand(x_0.shape[0], 1, 1, device=x_0.device) 
    x_noisy = (1-t)*x_0 + t*noise
    optimizer.zero_grad()
    x_pred = model(x_noisy)
    loss = F.mse_loss(x_pred, x_0)
    loss.backward()
    optimizer.step()
    return loss.item()