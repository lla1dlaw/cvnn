"""
Utilities for Federated ResNet Experiments using Flower.

This module contains helper functions for:
- Loading and partitioning the CIFAR-10 dataset for federated clients.
- Instantiating the correct ResNet model based on a configuration dictionary.
- Applying the specific weight initialization from the "Deep Complex Networks" paper.
- Defining the core training and testing logic that each client will execute.
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict
from tqdm import tqdm
import math
from torch.nn.init import orthogonal_

from residual import ComplexResNet, RealResNet, RealResidualBlock, ComplexResidualBlock
from complexPyTorch.complexLayers import ComplexConv2d, ComplexLinear

try:
    from torchmetrics.classification import (
        MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, 
        MulticlassF1Score, MulticlassAUROC
    )
except ImportError:
    print("Dependencies not found. Please install them: pip install torchmetrics")
    exit()

# --- WEIGHT INITIALIZATION (as per Paper Sec. 3.6) ---

def init_weights(m):
    """
    Applies the paper's weight initialization to the model's layers.
    """
    if isinstance(m, nn.Conv2d):
        fan_in = nn.init._calculate_fan_in_and_fan_out(m.weight)[0]
        flat_shape = (m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1])
        random_matrix = torch.randn(flat_shape)
        orthogonal_matrix = orthogonal_(random_matrix)
        reshaped_matrix = orthogonal_matrix.reshape(m.weight.shape)
        he_variance = 2.0 / fan_in
        scaling_factor = math.sqrt(he_variance * m.out_channels)
        with torch.no_grad():
            m.weight.copy_(reshaped_matrix * scaling_factor)
    elif isinstance(m, ComplexConv2d):
        real_conv = m.conv_r
        fan_in = real_conv.in_channels * real_conv.kernel_size[0] * real_conv.kernel_size[1]
        weight_shape = real_conv.weight.shape
        flat_shape = (weight_shape[0], weight_shape[1] * weight_shape[2] * weight_shape[3])
        random_matrix = torch.randn(flat_shape, dtype=torch.complex64)
        U, _, Vh = torch.linalg.svd(random_matrix, full_matrices=False)
        unitary_matrix_flat = U @ Vh
        unitary_matrix = unitary_matrix_flat.reshape(weight_shape)
        he_variance = 2.0 / fan_in
        scaling_factor = math.sqrt(he_variance * weight_shape[0])
        scaled_unitary = unitary_matrix * scaling_factor
        with torch.no_grad():
            m.conv_r.weight.copy_(scaled_unitary.real)
            m.conv_i.weight.copy_(scaled_unitary.imag)
    elif isinstance(m, ComplexLinear):
        real_fc = m.fc_r
        fan_in = real_fc.in_features
        random_matrix = torch.randn(real_fc.weight.shape, dtype=torch.complex64)
        U, _, Vh = torch.linalg.svd(random_matrix, full_matrices=False)
        unitary_matrix = U @ Vh
        he_variance = 2.0 / fan_in
        scaling_factor = math.sqrt(he_variance * real_fc.out_features)
        scaled_unitary = unitary_matrix * scaling_factor
        with torch.no_grad():
            m.fc_r.weight.copy_(scaled_unitary.real)
            m.fc_i.weight.copy_(scaled_unitary.imag)

# --- MODEL LOADING ---

def get_model(config):
    """Instantiates a model and applies paper-specific weight initialization."""
    model_type = config.get('model_type', 'Complex')
    arch = config.get('arch', 'WS')
    activation = config.get('activation', 'crelu')
    learn_imag = config.get('learn_imag', True)

    if model_type == 'Real':
        model = RealResNet(block_class=RealResidualBlock, architecture_type=arch)
    else:
        model = ComplexResNet(
            block_class=ComplexResidualBlock,
            architecture_type=arch,
            activation_function=activation,
            learn_imaginary_component=learn_imag
        )
    model.apply(init_weights)
    return model

# --- DATA LOADING & PARTITIONING ---

def get_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return trainset, testset

def load_partitioned_data(num_clients: int, batch_size: int):
    trainset, testset = get_datasets()
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = torch.utils.data.random_split(trainset, lengths, torch.Generator().manual_seed(42))
    trainloaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in datasets]
    # For federated simulation, client-side validation uses the global test set
    valloaders = [DataLoader(testset, batch_size=batch_size) for _ in range(num_clients)]
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloaders, valloaders, testloader

# --- METRICS, TRAINING & TESTING LOGIC ---

def get_metrics(device, num_classes=10):
    """Initializes and returns a dictionary of TorchMetrics metrics."""
    metrics = {
        "accuracy": MulticlassAccuracy(num_classes=num_classes, average='micro').to(device),
        "top_5_accuracy": MulticlassAccuracy(num_classes=num_classes, top_k=5).to(device),
        "precision_macro": MulticlassPrecision(num_classes=num_classes, average='macro').to(device),
        "recall_macro": MulticlassRecall(num_classes=num_classes, average='macro').to(device),
        "f1_score_micro": MulticlassF1Score(num_classes=num_classes, average='micro').to(device),
        "f1_score_macro": MulticlassF1Score(num_classes=num_classes, average='macro').to(device),
        "f1_score_weighted": MulticlassF1Score(num_classes=num_classes, average='weighted').to(device),
        "auroc": MulticlassAUROC(num_classes=num_classes, average="macro").to(device)
    }
    return metrics

def train(net, trainloader, epochs, device, learning_rate):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    net.train()
    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test(net, testloader, device):
    """Evaluate the network on the test set and return all metrics."""
    criterion = torch.nn.CrossEntropyLoss()
    metrics = get_metrics(device)
    loss = 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            probs = torch.softmax(outputs, dim=1)
            for name, metric in metrics.items():
                (metric.update(probs, targets) if name == 'auroc' else metric.update(outputs, targets))
    
    loss /= len(testloader.dataset)
    # Compute all metrics
    final_metrics = {key: metric.compute().item() for key, metric in metrics.items()}
    return loss, final_metrics

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
