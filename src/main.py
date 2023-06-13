from torchvision import datasets, transforms
from torch import nn
import torch
from VAE import VAE
from train import train
from plot import plot_reconstruction

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VAE().to(device)

mnist_data = datasets.MNIST('data', train=True, download=True,
                            transform=transforms.ToTensor())
train(model, mnist_data, epochs=10)

mnist_test_data = datasets.MNIST('data', train=False, transform=transforms.ToTensor())
plot_reconstruction(model, mnist_test_data, 10, "mnist")

fashion_data = datasets.FashionMNIST('data', train=True, download=True,
                                     transform=transforms.ToTensor())
train(model, fashion_data, epochs=10)

plot_reconstruction(model, mnist_test_data, 10, "fashionmnist")
