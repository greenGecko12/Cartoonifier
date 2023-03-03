# This scripts approximates any input face into StyleGAN's latent face


#!/bin/bash
python encode_image.py \
  --image_path /home/sai_k/DualStyleGAN/data/content/randomface.jpg \
  --dlatent_path /home/sai_k/DualStyleGAN/encoder/latent_codes/randomface.npy \ 
  --save_optimized_image True

# python encode_image.py --image_path /home/sai_k/DualStyleGAN/pics/female_face.jpg --dlatent_path /home/sai_k/DualStyleGAN/encoder/latent_codes/female_face.npy --save_optimized_image true
