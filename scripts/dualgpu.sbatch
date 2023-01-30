#!/bin/bash
#
#SBATCH --job-name=example-gpu # Job name for tracking
#SBATCH --partition=gpu-batch  # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6      # Number of CPU threads used by your job
#SBATCH --mem=60000            # RAM in MB needed
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

## Execute your program(s) ##
## A Python program requiring CUDA:
source /etc/profile.d/modules.sh
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
module load cuda10.1

## activating the conda virtual environment
conda activate dualstylegan_env

## inferencing
python style_transfer.py --style cartoon --style_id 10