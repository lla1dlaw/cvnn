""" 
This script implements a complex-valued neural network using PyTorch. 
Supporting Paper: https://doi.org/10.1103/PhysRevX.11.021060


Author: Liam Laidlaw
Date: 2023-10-30
filename: main.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import complextorch
from complextorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complextorch.complexFunctions import complex_relu, complex_max_pool2d
from complextorch.complexMetrics import ComplexCategoricalAccuracy

from ComplexNet import ComplexNet
import numpy as np
import tensorflow as tf


def load_complex_dataset(x_train, x_test):
    """Loads the MNIST dataset and applies the 2D Discrete Fourier Transform (DFT) to each image.
    Args:
        x_train (numpy.ndarray): The training images, shape (num_samples, 28, 28).
        x_test (numpy.ndarray): The test images, shape (num_samples, 28, 28).
    returns: A tuple containing the transformed training and test datasets.
    """

    x_train_complex = []
    x_test_complex = []
    for train_sample in x_train:
        # Apply the 2D Discrete Fourier Transform
        train_complex_image = np.fft.fft2(train_sample)
        x_train_complex.append(train_complex_image)
    for test_sample in x_test:
        # Apply the 2D Discrete Fourier Transform
        test_complex_image = np.fft.fft2(test_sample)
        x_test_complex.append(test_complex_image)
        
    out = np.array(x_train_complex).astype(np.complex64), np.array(x_test_complex).astype(np.complex64)
    return out

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).type(torch.complex64), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data), 
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader), 
                loss.item())
            )

def main():

    batch_size = 64
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = datasets.MNIST('../data', train=True, transform=trans, download=True)
    test_set = datasets.MNIST('../data', train=False, transform=trans, download=True)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size= batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size= batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ComplexNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Run training on 50 epochs
    for epoch in range(50):
        train(model, device, train_loader, optimizer, epoch)
    
    # test the model
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).type(torch.complex64), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() 


if __name__ == '__main__':
    main()
