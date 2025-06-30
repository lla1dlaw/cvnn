#!/bin/bash

#==============================================================================
# SLURM Submission Script for PyTorch ResNet Comparison (Maximized Resources)
#
# This script is designed to run the Python training script on a SLURM-managed
# cluster. It has been updated to request powerful GPU resources from the
# 'gpu' partition for the maximum allowed time.
#
# To submit this job, save it as 'run_training_max.sbatch' and run:
# sbatch run_training_max.sbatch
#==============================================================================

# --- SLURM JOB PARAMETERS ---

# Job name
#SBATCH --job-name=resnet_comparison_max

# Output and error files. %j will be replaced by the job ID.
#SBATCH --output=resnet_comparison.out
#SBATCH --error=resnet_comparison.err

# Resource requests - Maximized for the V100 GPU nodes
#SBATCH --partition=gpu             # Request the partition with V100 GPUs
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks (usually 1 for a single script)
#SBATCH --cpus-per-task=16          # Increased CPUs for data loading and processing
#SBATCH --gres=gpu:4                # Request 4 GPUs on a node
#SBATCH --mem=64G                   # Increased memory request (64 GB)
#SBATCH --time=1-00:00:00           # Set time limit to the 1-day maximum for this partition

# Email notifications for job status
#SBATCH --mail-type=ALL             # Send email on job start, end, and failure
#SBATCH --mail-user=liamlaidlaw04@gmail.com # <<< REPLACE WITH YOUR EMAIL

# --- JOB EXECUTION ---
./cleanup.sh
# Print job information
echo "Starting job $SLURM_JOB_ID on host $HOSTNAME"
echo "Job allocated to partition ${SLURM_JOB_PARTITION}"
echo "Job allocated ${SLURM_CPUS_ON_NODE} CPUs and ${SLURM_MEM_PER_NODE} MB of memory"
echo "Job allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "------------------------------------------------------"

module load cudnn8.5-cuda11.7/8.5.0.96
echo "CUDA/cuDNN module loaded."

mamba init bash
mamba activate torch_cvnn
echo "Activated Mamba environment: $CONDA_DEFAULT_ENV"

echo "Starting Python training script..."
python train_resnets.py \
  --epochs 50 \
  --batch-size 128 \
  --folds 1

echo "------------------------------------------------------"
echo "Job finished with exit code $? at $(date)"
