import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm, trange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#@title A handy training function
def train_diffusion_model(dataset,
                          score_model,
                          marginal_prob_std_fn,
                          n_epochs =   100,
                          batch_size =  1024,
                          lr=10e-4,
                          model_name="transformer"):
    # Print model architecture size
    total_params = sum(p.numel() for p in score_model.parameters())
    trainable_params = sum(p.numel() for p in score_model.parameters() if p.requires_grad)
    
    print(f"Model: {model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("-----------------------------")

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = Adam(score_model.parameters(), lr=lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.2, 0.98 ** epoch))
    tqdm_epoch = trange(n_epochs)
    for epoch in tqdm(tqdm_epoch):
        avg_loss = 0.
        num_items = 0
        for x, y in tqdm(data_loader):
            x = x.to(device)
            # if "ldm" in model_name:
            #     loss = loss_fn_cond_ldm(score_model, x, y, marginal_prob_std_fn)
            # else:
            loss = loss_fn_cond(score_model, x, y, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {loss:5f}")
    scheduler.step()
    lr_current = scheduler.get_last_lr()[0]
    print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, avg_loss / num_items, lr_current))
    # Print the averaged training loss so far.
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
    # Update the checkpoint after each epoch of training.
    torch.save(score_model.state_dict(), f'ckpt_{model_name}.pth')
    
class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.dense(x)[..., None, None]

def marginal_prob_std(t, sigma):
    t = t.to(device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
    return sigma**t.to(device)

import torch

def loss_fn_cond(model, x, y, marginal_prob_std, eps=1e-5):
    """
    Computes the loss for a conditional denoising diffusion probabilistic model (DDPM).

    Args:
        model: The neural network model that predicts the score (i.e., the gradient of the log probability).
        x (torch.Tensor): The original data samples (e.g., images) with shape (batch_size, channels, height, width).
        y (torch.Tensor): The conditional information (e.g., class labels or other auxiliary data).
        marginal_prob_std (function): A function that returns the standard deviation of the noise at a given time step.
        eps (float, optional): A small value to ensure numerical stability. Default is 1e-5.

    Returns:
        torch.Tensor: The computed loss as a scalar tensor.
    """
    
    # Sample a random time step for each sample in the batch.
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    
    # Sample random noise from a standard normal distribution with the same shape as the input.
    z = torch.randn_like(x)
    
    # Compute the standard deviation of the noise at the sampled time step.
    std = marginal_prob_std(random_t)
    
    # Perturb the input data with the sampled noise, scaled by the computed standard deviation.
    perturbed_x = x + z * std[:, None, None, None]
    
    # Predict the score (denoising direction) using the model.
    # The model takes the perturbed data, the time step, and the conditional information as inputs.
    score = model(perturbed_x, random_t, y=y)
    
    # Compute the loss as the mean squared error between the predicted score and the true noise,
    # weighted by the standard deviation.
    #loss = torch.mean(torch.sum((score * std[:, None, None, None] - z)**2, dim=(1,2,3)))
    loss = F.mse_loss(score * std[:, None, None, None], -z, reduction='mean')
    
    return loss

from torchvision.models import vgg16
import lpips

def loss_fn_cond_lpips(model, x, y, marginal_prob_std, eps=1e-5, lpips_weight=0.1):
    """
    Computes the loss for a conditional denoising diffusion probabilistic model (DDPM)
    with additional LPIPS perceptual loss.

    Args:
        model: The neural network model that predicts the score.
        x (torch.Tensor): The original data samples.
        y (torch.Tensor): The conditional information.
        marginal_prob_std (function): A function that returns the standard deviation of the noise.
        eps (float, optional): A small value to ensure numerical stability. Default is 1e-5.
        lpips_weight (float, optional): Weight for the LPIPS loss. Default is 0.1.

    Returns:
        torch.Tensor: The computed loss as a scalar tensor.
    """
    
    # Initialize LPIPS loss function
    lpips_fn = lpips.LPIPS(net='vgg').to(x.device)
    
    # Sample random time step and noise
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    z = torch.randn_like(x)
    
    # Compute standard deviation and perturb input
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
    
    # Predict the score
    score = model(perturbed_x, random_t, y=y)
    
    # Compute MSE loss
    mse_loss = F.mse_loss(score * std[:, None, None, None], -z, reduction='mean')
    
    # Compute LPIPS loss
    # We need to ensure the input is in the right format for LPIPS (3 channels)
    x_3ch = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
    perturbed_x_3ch = perturbed_x.repeat(1, 3, 1, 1) if perturbed_x.shape[1] == 1 else perturbed_x
    lpips_loss = lpips_fn(x_3ch, perturbed_x_3ch).mean()
    
    # Combine losses
    total_loss = mse_loss + lpips_weight * lpips_loss
    
    return total_loss


def Euler_Maruyama_sampler(score_model,
              marginal_prob_std,
              diffusion_coeff,
              num_steps,
              batch_size=64,
              x_shape=(1, 28, 28),
              device='cuda',
              eps=1e-3, y=None):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps.
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

    Returns:
    Samples.
    """
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, *x_shape, device=device) \
    * marginal_prob_std(t)[:, None, None, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step, y=y) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
    # Do not include any noise in the last sampling step.
    return mean_x