#!/bin/bash
#
#SBATCH --job-name=CartoonGAN # Job name for tracking
#SBATCH --partition=gpu-batch     # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6     # Number of CPU threads used by your job
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --output=./new_logs/joboutput_%j.out # Standard out from your job
#SBATCH --error=./new_logs/joboutput_%j.err  # Standard error from your job

## Execute your program(s) ##
## A Python program requiring CUDA:
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
export CXX=g++
export CUDA_HOME=/local/java/cuda-11.0

## activating the conda virtual environment
conda activate dualstylegan_env
module load CUDA/11.6.0

LATENT_CODE_NUM=1
python facial_editing/edit.py \
    -m stylegan_celebahq \
    -b facial_editing/boundaries/stylegan_celebahq_pose_boundary.npy \
    -n "$LATENT_CODE_NUM" \
    -o facial_editing/results/stylegan_celebahq_pose_editing