#!/bin/bash
#SBATCH -J NVIDIASMI        # job name
#SBATCH -o log.txt    # output and error file name (%j expands to jobID)
#SBATCH -c 48                # cpus per task
#SBATCH -N 1                # number of nodes you want to run on
#SBATCH --gres=gpu:4       # request a gpu
#SBATCH -p gpu			# queue (partition)
#SBATCH -t 72:00:00         # run time (hh:mm:ss)

module load conda
module load cudnn8.0-cuda11.0/8.0.5.39
conda init bash
conda activate tf_net_3.10
python main.py
