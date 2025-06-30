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
#SBATCH --job-name=resnet_comparison

# Output and error files. These will be overwritten on each run.
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
#SBATCH --mail-user=liamlaidlaw@boisestate.edu  # Email for SLURM notifications

# --- JOB EXECUTION ---

# Print job information
echo "Starting job $SLURM_JOB_ID on host $HOSTNAME"
echo "Job allocated to partition ${SLURM_JOB_PARTITION}"
echo "Job allocated ${SLURM_CPUS_ON_NODE} CPUs and ${SLURM_MEM_PER_NODE} MB of memory"
echo "Job allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "------------------------------------------------------"

# 1. Purge modules and load the Anaconda module
module purge
module load anaconda3

# 2. Load the specific CUDA/cuDNN module FIRST (CRITICAL STEP)
#    This sets up the environment variables (like LD_LIBRARY_PATH) correctly.
module load cudnn8.5-cuda11.7/8.5.0.96
echo "CUDA/cuDNN module loaded."

# 3. Activate your Conda environment using 'source activate' SECOND
#    The conda environment will now inherit the CUDA paths.
#    Replace 'torch_cvnn' with the name of your actual Conda environment.
source activate torch_cvnn
echo "Activated Conda environment: $CONDA_DEFAULT_ENV"

# 4. Diagnostic checks:
#    If nvidia-smi works but PyTorch shows CUDA as False, the problem is
#    that PyTorch was installed as a CPU-only build inside your conda env.
echo "--- Running Diagnostics ---"
echo "--- Checking GPU Driver with nvidia-smi ---"
nvidia-smi
echo "---"
echo "LD_LIBRARY_PATH is: $LD_LIBRARY_PATH"
echo "Which python: $(which python)"
echo "Verifying PyTorch CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'PyTorch CUDA version: {torch.version.cuda}'); print(f'Number of GPUs: {torch.cuda.device_count()}')"
echo "------------------------------------------------------"

# 5. Navigate to the directory containing your script
#    This line is optional if you submit the job from the correct directory.
# cd /path/to/your/project/directory

# 6. Run the Python training script
#    - The script will use all allocated GPUs automatically.
#    - Email notifications are now handled via a .env file.
echo "Starting Python training script..."
python train_resnets.py \
  --epochs 50 \
  --batch-size 128 \
  --folds 1

echo "------------------------------------------------------"
echo "Job finished with exit code $? at $(date)"
