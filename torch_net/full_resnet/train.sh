#!/bin/bash

#==============================================================================
# SLURM Submission Script for DataParallel ResNet Training (Borah HPC Optimized)
#
# This script is optimized to request powerful resources on a
# V100 node in the 'gpu' partition on the Borah cluster. It uses a standard
# python command to launch the DataParallel-enabled training script.
#
# Usage:
#   sbatch --export=BATCH_SIZE=128 train.sh
#==============================================================================

# --- SLURM JOB PARAMETERS (MAXIMIZED FOR BORAH V100 NODE) ---

# Job name
#SBATCH --job-name=dp_resnet_comparison

#SBATCH --output=dp_resnet.out
#SBATCH --error=dp_resnet.err

# Resource requests - Maximized for a l40 GPU node
#SBATCH --partition=gpu-l40             # Request the partition with V100 GPUs
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # A single main task
#SBATCH --cpus-per-task=48          # Request all 40 CPUs on the node
#SBATCH --gres=gpu:1                # Request all 4 GPUs on the node
#SBATCH --mem=180G                  # Request most of the memory (180 of 192 GB)
#SBATCH --time=7-00:00:00           # Set time limit to the 7-day maximum

# Email notifications for job status
#SBATCH --mail-type=ALL             # Send email on job start, end, and failure
#SBATCH --mail-user=liamlaidlaw@boisestate.edu

# --- JOB EXECUTION ---

# Print job information
echo "======================================================"
echo "Starting job $SLURM_JOB_ID on host $HOSTNAME"
echo "Job allocated to partition: ${SLURM_JOB_PARTITION}"
echo "Job allocated CPUs: ${SLURM_CPUS_ON_NODE}"
echo "Job allocated Memory: ${SLURM_MEM_PER_NODE} MB"
echo "Job allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Batch Size: ${BATCH_SIZE:=128}" # Default to 128 if not set
echo "======================================================"

# 1. Purge modules and load Conda/CUDA
module purge
module load conda
module load cudnn8.5-cuda11.7/8.5.0.96
echo "Modules loaded."

# 2. Activate your Conda environment
source activate torch_cvnn
echo "Activated Conda environment: $CONDA_DEFAULT_ENV"

# 3. Diagnostic checks
echo "--- Running Diagnostics ---"
nvidia-smi
echo "Which python: $(which python)"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'PyTorch version: {torch.__version__}'); print(f'GPU Count: {torch.cuda.device_count()}')"
echo "---------------------------"

# 4. Run the DataParallel training script
echo "Starting DataParallel training script..."
python train_resnets.py \
  --epochs 200 \
  --batch-size ${BATCH_SIZE} \
  --folds 1 \
  --arch WS \
  --act crelu \
  --learn_imag_mode true_only

echo "======================================================"
echo "Job finished with exit code $? at $(date)"
echo "======================================================"
