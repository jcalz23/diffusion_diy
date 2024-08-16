import torch
import functools
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from models import Latent_UNet_Tranformer
from utils import marginal_prob_std, diffusion_coeff, train_diffusion_model

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    # Define noise fns, params
    sigma = 25.0
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)

    # Init model
    print("initialize new score model...")
    score_model = torch.nn.DataParallel(
        Latent_UNet_Tranformer(
            marginal_prob_std=marginal_prob_std_fn, channels = [16, 32, 64, 128]
        )
    )
    score_model = score_model.to(device)

    # Define training params
    n_epochs = 100
    batch_size = 1024
    lr = 1e-3

    # Load the TensorDataset
    dataset = torch.load(f'latent_data/mnist_latent_16d.pt')

    # Run
    train_diffusion_model(dataset,
                        score_model,
                        marginal_prob_std_fn,
                        n_epochs=n_epochs,
                        batch_size=batch_size,
                        lr=lr,
                        model_name=f"mnist_ldm_{str(lr)}_{n_epochs}e")