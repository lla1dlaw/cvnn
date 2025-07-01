"""
PyTorch Training Script for Comparing a Specific Real and Complex ResNet

This script is configured to run a focused comparison between two models based
on the 'WS' architecture described in "Deep Complex Networks" (Trabelsi et al., 2018):
1. A Complex 'WS' ResNet using CReLU and a learned imaginary component.
2. An equivalent Real 'WS' ResNet for direct comparison.

It automatically adapts to the execution environment: DDP, single GPU, or CPU.

Usage:
    # To run with DDP on a multi-GPU machine (e.g., 2 GPUs):
    torchrun --nproc_per_node=2 train_resnets.py

    # To run on a single GPU or CPU (the script will auto-detect):
    python train_resnets.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import csv
import logging
import argparse
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import traceback
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from residual import ComplexResNet, RealResNet, RealResidualBlock, ComplexResidualBlock

try:
    from torchmetrics.classification import (
        MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, 
        MulticlassF1Score, MulticlassAUROC
    )
    from sklearn.model_selection import StratifiedKFold
except ImportError:
    print("Dependencies not found. Please install them: pip install torchmetrics scikit-learn python-dotenv tqdm")
    exit()

# --- DDP and Utility FUNCTIONS ---

def setup_ddp():
    """Initializes the DDP process group if in a DDP environment."""
    if 'WORLD_SIZE' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    """Cleans up the DDP process group if it was initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """Checks if the current process is the main one (rank 0) or if not in DDP."""
    return not dist.is_initialized() or dist.get_rank() == 0

def setup_logging():
    if is_main_process():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - RANK:%(rank)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
        )

def log_info(message):
    if is_main_process():
        rank = dist.get_rank() if dist.is_initialized() else 0
        adapter = logging.LoggerAdapter(logging.getLogger(), {'rank': rank})
        adapter.info(message)

def send_email_notification(subject, body):
    if not is_main_process(): return
    load_dotenv()
    email_user = os.getenv("EMAIL_USER")
    email_pass = os.getenv("EMAIL_PASS")
    if not email_user or not email_pass:
        log_info("WARNING: EMAIL_USER or EMAIL_PASS not found in .env file. Skipping email notification.")
        return
    msg = MIMEText(body)
    msg['Subject'] = f"[ResNet Training] {subject}"
    msg['From'] = email_user
    msg['To'] = email_user
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
            smtp_server.login(email_user, email_pass)
            smtp_server.sendmail(email_user, email_user, msg.as_string())
        log_info(f"Successfully sent notification email to {email_user}")
    except Exception as e:
        log_info(f"ERROR: Failed to send email: {e}")

def save_model(model, config, directory="saved_models", fold=None):
    if not is_main_process(): return
    if not os.path.exists(directory): os.makedirs(directory)
    fold_str = f"_fold{fold}" if fold is not None else ""
    filename = f"{config['name']}{fold_str}.pth"
    filepath = os.path.join(directory, filename)
    state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    torch.save(state_dict, filepath)
    log_info(f"SUCCESS: Model state_dict saved to {filepath}")
    return filepath

def save_results_to_csv(results, filename="training_results.csv"):
    if not is_main_process(): return
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = list(results.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
    log_info(f"SUCCESS: Results saved to {filename}")

# --- DATA AND METRICS SETUP ---

def get_datasets():
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
    if is_main_process(): # Download only on one process
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    if dist.is_initialized():
        dist.barrier() # Wait for main process to download
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    if is_main_process():
        log_info("CIFAR-10 datasets loaded successfully.")
    return trainset, testset

def get_metrics(device, num_classes=10):
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

def run_experiment_fold(config, args, train_loader, val_loader, fold_num, device, processing_mode):
    use_sync_bn = (processing_mode == 'DDP')
    
    if config['model_type'] == 'Real':
        model = RealResNet(block_class=RealResidualBlock, use_sync_bn=use_sync_bn, num_classes=10)
    else:
        model = ComplexResNet(block_class=ComplexResidualBlock, activation_function=config['activation'], learn_imaginary_component=config['learn_imag'], use_sync_bn=use_sync_bn, num_classes=10)
    
    model.to(device)
    if processing_mode == 'DDP':
        model = DDP(model, device_ids=[device], find_unused_parameters=False)
    
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True)
    criterion = nn.CrossEntropyLoss()
    train_accuracy_metric = MulticlassAccuracy(num_classes=10, average='micro').to(device)
    val_accuracy_metric = MulticlassAccuracy(num_classes=10, average='micro').to(device)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(args.epochs):
        if processing_mode == 'DDP':
            train_loader.sampler.set_epoch(epoch)
        
        log_info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        model.train()
        train_loss = 0.0
        train_accuracy_metric.reset()
        train_loop = tqdm(train_loader, desc=f"Training  ", leave=False, disable=not is_main_process())
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
            if is_main_process():
                train_loop.set_postfix(loss=f"{train_loss / (train_loop.n + 1):.4f}", acc=f"{train_accuracy_metric.compute().item():.4f}")
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_accuracy_metric.compute().item())

        model.eval()
        val_loss = 0.0
        val_accuracy_metric.reset()
        val_loop = tqdm(val_loader, desc=f"Validating", leave=False, disable=not is_main_process())
        with torch.no_grad():
            for inputs, targets in val_loop:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_accuracy_metric.update(outputs, targets)
                if is_main_process():
                    val_loop.set_postfix(loss=f"{val_loss / (val_loop.n + 1):.4f}", acc=f"{val_accuracy_metric.compute().item():.4f}")

        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_accuracy_metric.compute().item())
        
        log_info(f"Epoch {epoch+1} Summary | Train Loss: {history['train_loss'][-1]:.4f} | Train Acc: {history['train_acc'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f} | Val Acc: {history['val_acc'][-1]:.4f}")

    final_metrics = {}
    if is_main_process():
        metrics = get_metrics(device)
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model.module(inputs) if isinstance(model, DDP) else model(inputs)
                probs = torch.softmax(outputs, dim=1)
                for name, metric in metrics.items():
                    (metric.update(probs, targets) if name == 'auroc' else metric.update(outputs, targets))
        final_metrics = {key: metric.compute().item() for key, metric in metrics.items()}
        log_info(f"SUCCESS: Final Metrics for Fold {fold_num}: {final_metrics}")
        
    return model, final_metrics, history

# --- MAIN DRIVER ---
def main(args):
    use_ddp = 'WORLD_SIZE' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1
    if use_ddp:
        setup_ddp()
        processing_mode = 'DDP'
        device = int(os.environ["LOCAL_RANK"])
    elif torch.cuda.is_available():
        processing_mode = 'Single GPU'
        device = torch.device("cuda:0")
    else:
        processing_mode = 'CPU'
        device = torch.device("cpu")

    setup_logging()
    
    log_info(f"Starting training process using: {processing_mode}")
    
    experiment_configs = [
        {'name': 'Complex_ResNet_WS_CReLU_LearnImag', 'model_type': 'Complex', 'activation': 'crelu', 'learn_imag': True},
        {'name': 'Real_ResNet_WS_Equivalent', 'model_type': 'Real', 'activation': 'relu', 'learn_imag': 'N/A'}
    ]

    train_dataset, test_dataset = get_datasets()

    for config in experiment_configs:
        start_time = datetime.now()
        log_info("\n" + "="*80)
        log_info(f"STARTING NEW EXPERIMENT: {config['name']}")
        log_info(f"  - Processing Mode: {processing_mode}")
        log_info(f"  - Model Type: {config['model_type']}")
        log_info(f"  - Activation: {config['activation']}")
        log_info(f"  - Learn Imaginary: {config['learn_imag']}")
        log_info(f"  - Epochs: {args.epochs}")
        log_info(f"  - Batch Size: {args.batch_size}")
        log_info(f"  - Learning Rate: {args.learning_rate}")
        log_info(f"  - Folds: {args.folds}")
        log_info("="*80)
        
        try:
            fold_results = []
            if args.folds > 1:
                targets = np.array(train_dataset.targets)
                skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
                splits = list(skf.split(np.zeros(len(targets)), targets))
            else:
                splits = [(np.arange(len(train_dataset)), np.arange(len(test_dataset)))]
            
            for fold, (train_idx, val_idx) in enumerate(splits):
                fold_num = fold + 1
                train_subset = Subset(train_dataset, train_idx)
                if args.folds > 1:
                    val_dataset_no_aug = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=get_datasets()[1].transform)
                    val_subset = Subset(val_dataset_no_aug, val_idx)
                else:
                    val_subset = test_dataset
                
                train_sampler = DistributedSampler(train_subset) if use_ddp else None
                shuffle = not use_ddp

                train_loader = DataLoader(train_subset, batch_size=args.batch_size, sampler=train_sampler, shuffle=shuffle, num_workers=2, pin_memory=True)
                val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

                model, metrics, history = run_experiment_fold(config, args, train_loader, val_loader, fold_num, device, processing_mode)
                
                if use_ddp: dist.barrier()
                
                if is_main_process():
                    model_path = save_model(model, config, fold=fold_num)
                    fold_data = {'fold': fold_num, 'model_path': model_path, **metrics}
                    if args.folds <= 1: 
                        for i in range(args.epochs):
                            fold_data[f"epoch_{i+1}_train_loss"] = history['train_loss'][i]
                            fold_data[f"epoch_{i+1}_train_acc"] = history['train_acc'][i]
                            fold_data[f"epoch_{i+1}_val_loss"] = history['val_loss'][i]
                            fold_data[f"epoch_{i+1}_val_acc"] = history['val_acc'][i]
                    fold_results.append(fold_data)
                    log_info(f"SUCCESS: Fold {fold_num} for experiment {config['name']} completed.")

            if is_main_process():
                final_save_data = {'status': 'Completed', **config}
                if args.folds > 1:
                    metric_keys = list(fold_results[0].keys() - {'fold', 'model_path'})
                    for key in metric_keys:
                        values = [res[key] for res in fold_results]
                        final_save_data[f"{key}_mean"] = np.mean(values)
                        final_save_data[f"{key}_std"] = np.std(values)
                else:
                    final_save_data.update(fold_results[0])

                final_save_data['training_time_seconds'] = (datetime.now() - start_time).total_seconds()
                save_results_to_csv(final_save_data)
                log_info(f"SUCCESS: Experiment {config['name']} fully completed and results saved.")

        except Exception as e:
            if is_main_process():
                log_info(f"FAILURE: Experiment {config['name']} failed after {(datetime.now() - start_time).total_seconds():.2f} seconds.")
                log_info(traceback.format_exc())
                error_results = {**config, 'status': 'Failed', 'error': str(e)}
                save_results_to_csv(error_results)
                error_body = f"Experiment '{config['name']}' failed.\n\nError:\n{e}\n\nTraceback:\n{traceback.format_exc()}"
                send_email_notification("Training Script ERROR", error_body)

    if use_ddp:
        cleanup_ddp()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch ResNet Comparison Training Script (Portable)")
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--folds', type=int, default=1, help='Number of K-Folds for cross-validation. Default is 1 (standard train/test split).')
    
    args = parser.parse_args()
    main(args)
