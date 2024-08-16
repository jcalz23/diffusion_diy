import torch
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from tqdm import tqdm
from models import AutoEncoder

if __name__=="__main__":
    # Prepare dataloader
    batch_size = 4096
    dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load model
    device = 'cuda'
    ckpt = torch.load('model_objects/ckpt_ae_16d_mse_100e.pth', map_location=device)
    ae_model = AutoEncoder([4, 8, 16]).cuda()
    ae_model = ae_model.to(device)
    ae_model.load_state_dict(ckpt)
    ae_model.requires_grad_(False)
    ae_model.eval()

    # Run
    zs = []
    ys = []
    for x, y in tqdm(data_loader):
        z = ae_model.encoder(x.to(device)).cpu()
        zs.append(z)
        ys.append(y)

    zdata = torch.cat(zs, )
    ydata = torch.cat(ys, )

    # Save original
    latent_dataset = TensorDataset(zdata, ydata)
    torch.save(latent_dataset, 'mnist_latent_16d.pt')
    print("TensorDataset saved.")

