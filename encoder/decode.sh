#!/bin/bash
python InterFaceGAN/edit.py \
    --model_name stylegan_ffhq \
    --output_dir /home/sai_k/DualStyleGAN/encoder/InterFaceGAN/results/test_encoder_4  \
    --boundary_path /home/sai_k/DualStyleGAN/encoder/InterFaceGAN/boundaries/stylegan_ffhq_gender_w_boundary.npy \
    --input_latent_codes_path /home/sai_k/DualStyleGAN/encoder/latent_codes/female_face.npy \
    --latent_space_type wp