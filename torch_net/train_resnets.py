"""
PyTorch Training Script for Comparing Real and Complex ResNets

This script is portable and automatically adapts to the execution environment:
- Multi-GPU training using torch.nn.parallel.DistributedDataParallel (DDP).
- Single-GPU training if DDP is not available.
- CPU training as a fallback.

It correctly selects the appropriate Batch Normalization layer (Sync or standard)
for the environment.

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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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
    if 'WORLD_SIZE' in os.environ:
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

# Other utility functions (send_email, save_model, save_results_to_csv) remain the same
# but their calls will be guarded by is_main_process() where appropriate.

def save_model(model, config, directory="saved_models", fold=None):
    if not is_main_process(): return
    if not os.path.exists(directory): os.makedirs(directory)
    fold_str = f"_fold{fold}" if fold is not None else ""
    filename = f"{config['name']}{fold_str}.pth"
    filepath = os.path.join(directory, filename)
    # Handle both DDP-wrapped and standard models
    state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    torch.save(state_dict, filepath)
    log_info(f"SUCCESS: Model state_dict saved to {filepath}")
    return filepath

def save_results_to_csv(results, filename="training_results.csv"):
    if not is_main_process(): return
    # ... (logic is the same)

def get_datasets():
    # ... (logic is the same)

def get_metrics(device, num_classes=10):
    # ... (logic is the same)

# --- CORE TRAINING LOGIC ---

def run_experiment_fold(config, args, train_loader, val_loader, fold_num, device, processing_mode):
    """Runs a single training and validation experiment for one fold."""
    use_sync_bn = (processing_mode == 'DDP')
    
    if config['model_type'] == 'Real':
        model = RealResNet(block_class=RealResidualBlock, architecture_type=config['arch'], use_sync_bn=use_sync_bn, num_classes=10)
    else:
        model = ComplexResNet(block_class=ComplexResidualBlock, activation_function=config['activation'], architecture_type=config['arch'], learn_imaginary_component=config['learn_imag'], use_sync_bn=use_sync_bn, num_classes=10)
    
    model.to(device)
    if processing_mode == 'DDP':
        model = DDP(model, device_ids=[device])
    
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True)
    criterion = nn.CrossEntropyLoss()
    # ... (metrics setup and training loop remain largely the same) ...
    
    for epoch in range(args.epochs):
        if processing_mode == 'DDP':
            train_loader.sampler.set_epoch(epoch)
        # ...
    
    # Final evaluation
    final_metrics = {}
    # ... (evaluation logic remains the same, but only run on main process for DDP)
    if is_main_process():
        # ...
        log_info(f"SUCCESS: Final Metrics for Fold {fold_num}: {final_metrics}")
        
    return model, final_metrics, {}

# --- MAIN DRIVER ---
def main(args):
    # Determine processing mode
    use_ddp = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
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
    
    # ... (experiment config setup is the same) ...
    
    train_dataset, test_dataset = get_datasets()

    for config in experiment_configs:
        start_time = datetime.now()
        log_info("\n" + "="*80)
        log_info(f"STARTING NEW EXPERIMENT: {config['name']}")
        log_info(f"  - Processing Mode: {processing_mode}")
        # ... (log other hyperparameters) ...
        
        try:
            fold_results = []
            if args.folds > 1:
                # ... (KFold setup is the same) ...
            else:
                splits = [(np.arange(len(train_dataset)), np.arange(len(test_dataset)))]
            
            for fold, (train_idx, val_idx) in enumerate(splits):
                # ...
                # Create sampler based on processing mode
                if processing_mode == 'DDP':
                    train_sampler = DistributedSampler(Subset(train_dataset, train_idx))
                    shuffle = False
                else:
                    train_sampler = SubsetRandomSampler(train_idx) if args.folds > 1 else None
                    shuffle = True if train_sampler is None else False

                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=shuffle, num_workers=2)
                # ...
                
                model, metrics, history = run_experiment_fold(config, args, train_loader, val_loader, fold+1, device, processing_mode)
                
                if use_ddp: dist.barrier()
                
                if is_main_process():
                    # ... (save model and results) ...
            
            if is_main_process():
                # ... (aggregate and save final results) ...

        except Exception as e:
            if is_main_process():
                # ... (error handling) ...

    if use_ddp:
        cleanup_ddp()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch ResNet Comparison Training Script (Portable)")
    # ... (args are the same) ...
    args = parser.parse_args()
    main(args)
