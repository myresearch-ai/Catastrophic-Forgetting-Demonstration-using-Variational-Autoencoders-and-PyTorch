import torch
import os  
from torchvision import utils  
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from VAE import VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_reconstruction(model, dataset, epoch, dataset_name):
    model.eval()
    test_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data.to(device)
        recon_batch, _, _ = model(data)

        n = min(data.size(0), 8)
        comparison = torch.cat([data[:n],
                                recon_batch.view(16, 1, 28, 28)[:n]])
        comparison = comparison.cpu()

        plt.imshow(utils.make_grid(comparison).numpy().transpose(1, 2, 0))

        # check if images directory exists, if not create it
        if not os.path.exists('../images'):
            os.makedirs('../images')

        # change saving path to the images directory
        plt.savefig(
            f'../images/reconstruction_{dataset_name}_epoch_{epoch}.png')
        plt.close()
