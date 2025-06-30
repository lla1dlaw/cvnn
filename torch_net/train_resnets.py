"""
PyTorch Training Script for Comparing Real and Complex ResNets

This script automates the training and evaluation of various ResNet architectures
defined in 'residual.py' to compare the performance of real and
complex-valued models.

It iterates through all specified permutations of architectures, activation functions,
and other settings, logging results and saving them to a CSV file.

Features:
- Multi-GPU training using torch.nn.DataParallel.
- Securely loads email credentials from a .env file.
- Optional K-Fold Cross-Validation for robust evaluation.
- Saves per-epoch training/validation history (loss, accuracy) for standard splits.
- Saves the trained weights (state_dict) for each successful experiment/fold.
- Uses CIFAR-10 dataset with standard data augmentation.
- Logs training progress to both console and a file ('training.log').
- Rich console logging with TQDM progress bars and live metrics.
- Calculates and reports a comprehensive set of classification metrics.
- Saves all experiment results to 'training_results.csv'.
- Optional email notifications for errors during training.

Usage:
    # To run this script, first create a .env file with your email credentials.
    # Then, install the required libraries:
    # pip install python-dotenv torchmetrics scikit-learn tqdm

    # Standard train/test split
    python train_resnets.py

    # With 5-Fold Cross-Validation on the training set
    python train_resnets.py --folds 5

Note on Cross-Validation: Using K-Fold CV will multiply the total training
time by K. It is recommended for final, robust evaluation of specific models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler

import os
import csv
import logging
import argparse
import smtplib
from email.mime.text import MIMEText
from itertools import product
from datetime import datetime
import traceback
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

# Make sure the models from the canvas are in this file or imported
# Assuming 'residual.py' and 'activations.py' are in the same directory.
from residual import ComplexResNet, RealResNet, RealResidualBlock, ComplexResidualBlock

# TorchMetrics for calculating classification metrics
try:
    from torchmetrics.classification import (
        MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, 
        MulticlassF1Score, MulticlassAUROC
    )
    from sklearn.model_selection import StratifiedKFold
except ImportError:
    print("Dependencies not found. Please install them: pip install torchmetrics scikit-learn python-dotenv tqdm")
    exit()

# --- UTILITY FUNCTIONS ---

def setup_logging():
    """Configures logging to output to both a file and the console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )

def send_email_notification(subject, body):
    """Sends an email notification using credentials from a .env file."""
    load_dotenv()
    email_user = os.getenv("EMAIL_USER")
    email_pass = os.getenv("EMAIL_PASS")

    if not email_user or not email_pass:
        logging.warning("EMAIL_USER or EMAIL_PASS not found in .env file. Skipping email notification.")
        return

    msg = MIMEText(body)
    msg['Subject'] = f"[ResNet Training] {subject}"
    msg['From'] = email_user
    msg['To'] = email_user
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
            smtp_server.login(email_user, email_pass)
            smtp_server.sendmail(email_user, email_user, msg.as_string())
        logging.info(f"Successfully sent notification email to {email_user}")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

def save_model(model, config, directory="saved_models", fold=None):
    """Saves the model's state dictionary to a file."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    fold_str = f"_fold{fold}" if fold is not None else ""
    filename = f"{config['name']}{fold_str}.pth"
    filepath = os.path.join(directory, filename)

    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), filepath)
    else:
        torch.save(model.state_dict(), filepath)

    logging.info(f"SUCCESS: Model state_dict saved to {filepath}")
    return filepath

def save_results_to_csv(results, filename="training_results.csv"):
    """Appends a dictionary of results to a CSV file."""
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = list(results.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
    logging.info(f"SUCCESS: Results saved to {filename}")


# --- DATA AND METRICS SETUP ---

def get_datasets():
    """Returns the CIFAR-10 train and test datasets with appropriate transforms."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    logging.info("CIFAR-10 datasets loaded successfully.")
    return trainset, testset

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

# --- CORE TRAINING LOGIC ---

def run_experiment_fold(config, args, train_loader, val_loader, fold_num, device):
    """Runs a single training and validation experiment for one fold."""
    
    # --- Model and Optimizer Setup ---
    if config['model_type'] == 'Real':
        model = RealResNet(block_class=RealResidualBlock, architecture_type=config['arch'], num_classes=10)
    else:
        model = ComplexResNet(block_class=ComplexResidualBlock, activation_function=config['activation'], architecture_type=config['arch'], learn_imaginary_component=config['learn_imag'], num_classes=10)
    
    model.to(device)
    if device == 'cuda' and torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs via DataParallel.")
        model = nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True)
    criterion = nn.CrossEntropyLoss()
    train_accuracy_metric = MulticlassAccuracy(num_classes=10, average='micro').to(device)
    val_accuracy_metric = MulticlassAccuracy(num_classes=10, average='micro').to(device)
    
    logging.info("Model and optimizer configured successfully.")
    
    # --- Per-Epoch History Tracking ---
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # --- Training & Validation Loop ---
    for epoch in range(args.epochs):
        logging.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_accuracy_metric.reset()
        train_loop = tqdm(train_loader, desc=f"Training  ", leave=False)
        for inputs, targets in train_loop:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_accuracy_metric.update(outputs, targets)
            
            # Update TQDM progress bar
            train_loop.set_postfix(
                loss=f"{train_loss / (train_loop.n + 1):.4f}", 
                acc=f"{train_accuracy_metric.compute().item():.4f}"
            )
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_accuracy_metric.compute().item())

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_accuracy_metric.reset()
        val_loop = tqdm(val_loader, desc=f"Validating", leave=False)
        with torch.no_grad():
            for inputs, targets in val_loop:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_accuracy_metric.update(outputs, targets)
        
                val_loop.set_postfix(
                    loss=f"{val_loss / (val_loop.n + 1):.4f}", 
                    acc=f"{val_accuracy_metric.compute().item():.4f}"
                )

        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_accuracy_metric.compute().item())
        
        logging.info(f"Epoch {epoch+1} Summary | Train Loss: {history['train_loss'][-1]:.4f} | Train Acc: {history['train_acc'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f} | Val Acc: {history['val_acc'][-1]:.4f}")

    # --- Final Evaluation on Validation Set ---
    metrics = get_metrics(device)
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            for name, metric in metrics.items():
                (metric.update(probs, targets) if name == 'auroc' else metric.update(outputs, targets))
    final_metrics = {key: metric.compute().item() for key, metric in metrics.items()}
    logging.info(f"SUCCESS: Final Metrics for Fold {fold_num}: {final_metrics}")
        
    return model, final_metrics, history

# --- MAIN DRIVER ---
def main(args):
    setup_logging()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Starting training process on device: {device}")
    
    arch_types, complex_activations, learn_imag_opts = ['WS', 'DN', 'IB'], ['crelu', 'zrelu', 'modrelu', 'complex_cardioid'], [True, False]
    experiment_configs = []

    for arch, act, learn in product(arch_types, complex_activations, learn_imag_opts):
        experiment_configs.append({'name': f"Complex_ResNet_{arch}_{act}_{'LearnImag' if learn else 'ZeroImag'}", 'model_type': 'Complex', 'arch': arch, 'activation': act, 'learn_imag': learn})
    for arch in arch_types:
        experiment_configs.append({'name': f"Real_ResNet_{arch}", 'model_type': 'Real', 'arch': arch, 'activation': 'relu', 'learn_imag': 'N/A'})

    train_dataset, test_dataset = get_datasets()

    for config in experiment_configs:
        start_time = datetime.now()
        
        # --- Log Experiment Hyperparameters ---
        logging.info("\n" + "="*80)
        logging.info(f"STARTING NEW EXPERIMENT: {config['name']}")
        logging.info(f"  - Model Type: {config['model_type']}")
        logging.info(f"  - Architecture: {config['arch']}")
        logging.info(f"  - Activation: {config['activation']}")
        logging.info(f"  - Learn Imaginary: {config['learn_imag']}")
        logging.info(f"  - Epochs: {args.epochs}")
        logging.info(f"  - Batch Size: {args.batch_size}")
        logging.info(f"  - Learning Rate: {args.learning_rate}")
        logging.info(f"  - Folds: {args.folds}")
        logging.info("="*80)

        try:
            fold_results = []
            
            if args.folds > 1:
                targets = np.array(train_dataset.targets)
                skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
                splits = skf.split(np.zeros(len(targets)), targets)
            else:
                splits = [(np.arange(len(train_dataset)), np.arange(len(test_dataset)))]
            
            for fold, (train_idx, val_idx) in enumerate(splits):
                fold_num = fold + 1
                
                if args.folds > 1:
                    train_sampler = SubsetRandomSampler(train_idx)
                    val_dataset_no_aug = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=get_datasets()[1].transform)
                    val_loader = DataLoader(Subset(val_dataset_no_aug, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=2)
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=2)
                else: 
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
                    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

                model, metrics, history = run_experiment_fold(config, args, train_loader, val_loader, fold_num, device)
                model_path = save_model(model, config, fold=fold_num)
                
                fold_data = {'fold': fold_num, 'model_path': model_path, **metrics}
                if args.folds <= 1: 
                    for i in range(args.epochs):
                        fold_data[f"epoch_{i+1}_train_loss"] = history['train_loss'][i]
                        fold_data[f"epoch_{i+1}_train_acc"] = history['train_acc'][i]
                        fold_data[f"epoch_{i+1}_val_loss"] = history['val_loss'][i]
                        fold_data[f"epoch_{i+1}_val_acc"] = history['val_acc'][i]
                fold_results.append(fold_data)
                logging.info(f"SUCCESS: Fold {fold_num} for experiment {config['name']} completed.")

            # --- Aggregate and Save Results ---
            final_save_data = {'status': 'Completed', **config}
            if args.folds > 1:
                metric_keys = metrics.keys()
                for key in metric_keys:
                    values = [res[key] for res in fold_results]
                    final_save_data[f"{key}_mean"] = np.mean(values)
                    final_save_data[f"{key}_std"] = np.std(values)
            else:
                final_save_data.update(fold_results[0])

            final_save_data['training_time_seconds'] = (datetime.now() - start_time).total_seconds()
            save_results_to_csv(final_save_data)
            logging.info(f"SUCCESS: Experiment {config['name']} fully completed and results saved.")

        except Exception as e:
            logging.error(f"FAILURE: Experiment {config['name']} failed after {(datetime.now() - start_time).total_seconds():.2f} seconds.")
            logging.error(traceback.format_exc())
            error_results = {**config, 'status': 'Failed', 'error': str(e)}
            save_results_to_csv(error_results)
            error_body = f"Experiment '{config['name']}' failed.\n\nError:\n{e}\n\nTraceback:\n{traceback.format_exc()}"
            send_email_notification("Training Script ERROR", error_body)

    logging.info("\n" + "="*80)
    logging.info("--- ALL EXPERIMENTS FINISHED ---")
    logging.info("="*80)
    send_email_notification("Training Complete", "All ResNet experiments have finished running.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch ResNet Comparison Training Script")
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--folds', type=int, default=1, help='Number of K-Folds for cross-validation. Default is 1 (standard train/test split).')
    
    args = parser.parse_args()
    main(args)
