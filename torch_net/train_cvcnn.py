"""
Author: Liam Laidlaw
Purpose: Training test script for the cvcnn class on MNIST 
"""
import pretty_errors
from cvcnn import MNIST_CVCNN 
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
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


def train_on_MNIST():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # training hyper parameters
    batch_size = 64
    lr = 0.01
    momentum = 0.9
    epochs = 3

    # load data
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = datasets.MNIST('./data', train=True, transform=trans, download=True)
    test_set = datasets.MNIST('./data', train=False, transform=trans, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size= batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size= batch_size, shuffle=True)

    # init model and optimizer
    model = MNIST_CVCNN(hidden_activation='crelu', output_activation=None)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # train model
    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, epoch)


def main():
    train_on_MNIST()


if __name__ == "__main__":
    main()
