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
    ckpt = torch.load('model_objects/ckpt_mnist_ae_50e.pth', map_location=device)
    ae_model = AutoEncoder([4, 4, 4]).cuda()
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
    torch.save(latent_dataset, 'mnist_latent_original.pt')
    print("TensorDataset saved.")

    # 2. Standardized version
    latent_data, labels = latent_dataset.tensors
    mean = latent_data.mean(dim=0, keepdim=True)
    std = latent_data.std(dim=0, keepdim=True)
    standardized_data = (latent_data - mean) / std
    standardized_dataset = torch.utils.data.TensorDataset(standardized_data, labels)
    torch.save(standardized_dataset, 'mnist_latent_standardized.pt')

    # 3. (Optional) Scaled version to [-1, 1]
    min_val = latent_data.min(dim=0, keepdim=True)[0]
    max_val = latent_data.max(dim=0, keepdim=True)[0]
    scaled_data = (latent_data - min_val) / (max_val - min_val) * 2 - 1
    scaled_dataset = torch.utils.data.TensorDataset(scaled_data, labels)
    torch.save(scaled_dataset, 'mnist_latent_scaled.pt')
    print("Datasets saved: original, standardized, and scaled.")
