#!/bin/bash
#
#SBATCH --job-name=CartoonGAN # Job name for tracking
#SBATCH --partition=gpu-batch     # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6     # Number of CPU threads used by your job
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

## Execute your program(s) ##
## A Python program requiring CUDA:
source /etc/profile.d/modules.sh
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

## activating the conda virtual environment
conda activate dualstylegan_env
module load CUDA/11.6.0

## inferencing
## python style_transfer.py --style cartoon --style_id 10
python -c "import torch; print(str(torch.version.cuda))"


