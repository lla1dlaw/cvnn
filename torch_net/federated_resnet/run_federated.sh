#!/bin/bash

#==============================================================================
# SLURM Submission Script for Federated ResNet Training
#
# This script runs the Flower-based federated learning simulation.
# Note: Flower's `start_simulation` typically uses a single process to
# simulate clients, so we request one powerful GPU for the entire job.
#
# Usage:
#   sbatch run_federated.sh
#==============================================================================

# --- SLURM JOB PARAMETERS ---

# Job name
#SBATCH --job-name=federated_resnet

# Output and error files
#SBATCH --output=federated_resnet.out
#SBATCH --error=federated_resnet.err

# Resource requests
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1  # Requesting a single GPU for the simulation
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00 # 2 days

# Email notifications
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liamlaidlaw@boisestate.edu

# --- JOB EXECUTION ---

echo "======================================================"
echo "Starting job $SLURM_JOB_ID on host $HOSTNAME"
echo "======================================================"

# 1. Load modules
module purge
module load conda
module load cudnn8.5-cuda11.7/8.5.0.96
echo "Modules loaded."

# 2. Activate Conda environment
source activate torch_cvnn
echo "Activated Conda environment: $CONDA_DEFAULT_ENV"

# 3. Diagnostic checks
echo "--- Running Diagnostics ---"
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}');"
echo "---------------------------"

# 4. Run the Federated Learning script
#    Pass all desired experiment parameters here.
echo "Starting Flower federated learning script..."
python federated.py \
  --num_clients 5 \
  --num_rounds 10 \
  --local_epochs 5 \
  --architectures WS \
  --activations crelu \
  --learn_imag_mode true_only

echo "======================================================"
echo "Job finished with exit code $? at $(date)"
echo "======================================================"
