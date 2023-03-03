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


# LATENT_CODE_NUM=1
# python facial_editing/edit.py \
#     -m pggan_celebahq \
#     -b facial_editing/boundaries/pggan_celebahq_smile_boundary.npy \
#     -n "$LATENT_CODE_NUM" \
#     -o facial_editing/results/pggan_celebahq_smile_editing

# python facial_editing/edit.py \
#     --model_name stylegan_celebahq \
#     --output_dir facial_editing/new_results/inputted_face \
#     --boundary_path facial_editing/boundaries/stylegan_celebahq_smile_w_boundary.npy \
#     --latent_space_type wp \
#     --input_latent_codes_path /dcs/20/u2004267/CS310/DualStyleGAN/facial_editing/data/instyle.npy


# python facial_editing/generate_data.py \
#     --model_name stylegan_ffhq  \
#     --output_dir facial_editing/new_results/test_encoder \
#     --latent_space_type wp \
#     --latent_codes_path /dcs/20/u2004267/CS310/DualStyleGAN/facial_editing/data/female_face.npy


# python facial_editing/generate_data.py --model_name stylegan_celebahq --output_dir facial_editing/new_results/encoder_test --num 1

# python facial_editing/edit.py \
#     --model_name stylegan_ffhq \
#     --output_dir facial_editing/new_results/test_encoder_4  \
#     --boundary_path facial_editing/boundaries/stylegan_ffhq_smile_w_boundary.npy \
#     --input_latent_codes_path /dcs/20/u2004267/CS310/DualStyleGAN/facial_editing/data/female_face.npy \
#     --latent_space_type wp

# --generate_style \


python facial_editing/edit.py \
    --model_name stylegan_ffhq \
    --output_dir facial_editing/new_results/test_encoder_6  \
    --boundary_path facial_editing/boundaries/stylegan_ffhq_smile_w_boundary.npy \
    --input_latent_codes_path /dcs/20/u2004267/CS310/DualStyleGAN/facial_editing/data/female_face.npy \
    --latent_space_type wp \
    --start_distance 0 \
    --end_distance 0 \
    --steps 1