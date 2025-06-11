""" 
This script implements a complex-valued neural network using PyTorch. 
Supporting Paper: https://doi.org/10.1103/PhysRevX.11.021060


Author: Liam Laidlaw
Date: 2023-10-30
filename: main.py
"""

import pretty_errors

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ComplexNet import ComplexNet
from complextorch.nn.modules.loss import CVCauchyError
import numpy as np
from tqdm import tqdm 


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
        
    out = torch.from_numpy(np.array(x_train_complex).astype(np.complex64).flatten()), torch.from_numpy(np.array(x_test_complex).astype(np.complex64).flatten())
    return out

def train(model, device, train_loader, optimizer, loss, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).type(torch.complex64), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data), 
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader), 
                loss.item()))
        return loss.item()

def main():
    # training params
    batch_size = 64
    epochs = 20
    loss = CVCauchyError  # complex-valued loss function

    # model params
    in_size = 28 * 28 
    out_size = 10
    linear_shape = [100, 100] # shape of hidden linear layers

    # Load MNIST datasets
    (real_images_train, labels_train), (real_images_test, labels_test) = tf.keras.datasets.mnist.load_data() # real data
    complex_images_train, complex_images_test = load_complex_dataset(real_images_train, real_images_test) # complex data (2d DFT)

    # generate datasets 
    real_train_dataset = TensorDataset(real_images_train, labels_train)
    real_test_dataset = TensorDataset(real_images_test, labels_test)
    complex_train_dataset = TensorDataset(complex_images_train, labels_train)
    complex_test_dataset = TensorDataset(complex_images_test, labels_test)

    # create dataloaders
    real_train_loader = DataLoader(real_train_dataset, batch_size=batch_size, shuffle=True)
    real_test_loader = DataLoader(real_test_dataset, batch_size=batch_size, shuffle=False)
    complex_train_loader = DataLoader(complex_train_dataset, batch_size=batch_size, shuffle=True)
    complex_test_loader = DataLoader(complex_test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ComplexNet(in_size, out_size, ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Run training on 50 epochs
    for epoch in tqdm(range(epochs)):
        train(model, device, complex_train_loader, optimizer, loss, epoch)
    
    # test the model
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in complex_test_loader:
            data, target = data.to(device).type(torch.complex64), target.to(device)
            output = model(data)
            test_loss += loss(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() 


if __name__ == '__main__':
    main()
