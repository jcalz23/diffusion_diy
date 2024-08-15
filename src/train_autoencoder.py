from lpips import LPIPS
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import trange
from torch.optim import Adam
from models import AutoEncoder

if __name__ == "__main__":
    # Define the loss function, MSE and LPIPS
    lpips = LPIPS(net="squeeze").cuda()
    loss_fn_ae = lambda x,xhat: nn.functional.mse_loss(x, xhat) + lpips(x.repeat(1,3,1,1), x_hat.repeat(1,3,1,1)).mean()

    ae_model = AutoEncoder([4, 4, 4]).cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs =   100  #@param {'type':'integer'}
    ## size of a mini-batch
    batch_size =  2048   #@param {'type':'integer'}
    ## learning rate
    lr=10e-4 #@param {'type':'number'}

    dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    total_params = sum(p.numel() for p in ae_model.parameters())
    trainable_params = sum(p.numel() for p in ae_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("-----------------------------")

    optimizer = Adam(ae_model.parameters(), lr=lr)
    tqdm_epoch = trange(n_epochs)
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x, y in data_loader:
            x = x.to(device)
            z = ae_model.encoder(x)
            x_hat = ae_model.decoder(z)
            loss = loss_fn_ae(x, x_hat) #loss_fn_cond(score_model, x, y, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
    print('{} Average Loss: {:5f}'.format(epoch, avg_loss / num_items))
    # Print the averaged training loss so far.
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
    # Update the checkpoint after each epoch of training.
    torch.save(ae_model.state_dict(), f'ckpt_mnist_new_ae_{n_epochs}e.pth')