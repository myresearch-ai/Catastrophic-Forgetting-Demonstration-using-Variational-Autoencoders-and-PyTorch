import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from loss import loss_function
from VAE import VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, dataset, epochs=10):
    optimizer = Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    model.train()

    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    loss.item() / len(data)))
