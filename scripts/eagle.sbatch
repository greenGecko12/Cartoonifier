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

## inferencing
## python style_transfer.py --style cartoon --style_id 10
# python -c "import torch; print(str(torch.version.cuda))"

python style_transfer.py --style cartoon --style_id 177

# Fine-tune StyleGAN. Fine-tune StyleGAN in distributed settings:
# python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 finetune_stylegan.py --iter 600 --batch 2 --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt --style cartoon --augment ./data/cartoon/lmdb/

# Destylize artistic portraits.
# python destylize.py --model_name finetune-000600.pt --batch 4 --iter 300 cartoon

# Stage 1 & 2: Pretrain DualStyleGAN on FFHQ (don't have the FFHQ data for this, ignore for now, just use their model)
# python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 pretrain_dualstylegan.py --iter 3000 --batch 4 ./data/ffhq/lmdb/

# Fine-Tune DualStyleGAN on Target Domain. Fine-tune DualStyleGAN in distributed settings
# python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 finetune_dualstylegan.py --iter 1500 --batch 2 --ckpt ./checkpoint/generator-pretrain.pt --style_loss 0.25 --CX_loss 0.25 --perc_loss 1 --id_loss 1 --L2_reg_loss 0.015 --augment cartoon 


## DON'T BOTHER WITH THE OPTIONAL BITS - IT MESSES WITH THE ORDERING OF THE REFERENCE STYLE IMAGES
# Refine extrinsic style code. Refine the color and structure styles to better fit the example style images.
# python refine_exstyle.py --lr_color 0.1 --lr_structure 0.005 --ckpt ./checkpoint/cartoon/generator-001400.pt cartoon
