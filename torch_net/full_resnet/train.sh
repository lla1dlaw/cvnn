#!/bin/bash

#==============================================================================
# SLURM Submission Script for DDP ResNet Training (Borah HPC Optimized)
#
# This script is optimized to request the maximum available resources on a
# V100 node in the 'gpu' partition on the Borah cluster. It uses torchrun
# to correctly launch the DistributedDataParallel training script.
#
# Usage:
#   # Submit a job with a per-GPU batch size of 128
#   sbatch --export=BATCH_SIZE=128 train_ddp.sh
#
#   # Submit a job with a different batch size
#   sbatch --export=BATCH_SIZE=256 train_ddp.sh
#==============================================================================

# --- SLURM JOB PARAMETERS (MAXIMIZED FOR BORAH V100 NODE) ---

# Job name
#SBATCH --job-name=ddp_resnet_comparison

# Output and error files. Using %j for job ID to avoid overwrites.
#SBATCH --output=ddp_resnet_%j.out
#SBATCH --error=ddp_resnet_%j.err

# Resource requests - Maximized for a V100 GPU node
#SBATCH --partition=gpu             # Request the partition with V100 GPUs
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # A single main task that will spawn DDP processes
#SBATCH --cpus-per-task=40          # Request all 40 CPUs on the node
#SBATCH --gres=gpu:4                # Request all 4 GPUs on the node
#SBATCH --mem=180G                  # Request most of the memory (180 of 192 GB)
#SBATCH --time=7-00:00:00           # Set time limit to the 7-day maximum

# Email notifications for job status
#SBATCH --mail-type=ALL             # Send email on job start, end, and failure
#SBATCH --mail-user=liamlaidlaw@boisestate.edu

# --- JOB EXECUTION ---

./cleanup.sh

# Print job information
echo "======================================================"
echo "Starting job $SLURM_JOB_ID on host $HOSTNAME"
echo "Job allocated to partition: ${SLURM_JOB_PARTITION}"
echo "Job allocated CPUs: ${SLURM_CPUS_ON_NODE}"
echo "Job allocated Memory: ${SLURM_MEM_PER_NODE} MB"
echo "Job allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Per-GPU Batch Size: ${BATCH_SIZE:=128}" # Default to 128 if not set
echo "======================================================"

# 1. Set environment variables
# Set the number of GPUs for torchrun
export NGPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NGPUS GPUs."

# Ensure CUDA devices are visible to PyTorch
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 2. Purge modules and load Conda/CUDA
module purge
module load conda # Corrected to use the 'conda' module
module load cudnn8.5-cuda11.7/8.5.0.96
echo "Modules loaded."

# 3. Activate your Conda environment
# Replace 'torch_cvnn' with the name of your actual Conda environment.
source activate torch_cvnn
echo "Activated Conda environment: $CONDA_DEFAULT_ENV"

# 4. Diagnostic checks
echo "--- Running Diagnostics ---"
nvidia-smi
echo "Which python: $(which python)"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'PyTorch version: {torch.__version__}')"
echo "---------------------------"

# 5. Run the DDP training script using torchrun
#    - torchrun handles setting the environment variables for each process.
#    - --nproc_per_node should match the number of GPUs requested (--gres=gpu:N)
echo "Starting DDP training script with torchrun..."
torchrun --nproc_per_node=$NGPUS train_resnets.py \
  --epochs 50 \
  --batch-size ${BATCH_SIZE} \
  --folds 1

echo "======================================================"
echo "Job finished with exit code $? at $(date)"
echo "======================================================"
