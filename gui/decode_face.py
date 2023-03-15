from dualstylegan import Model
import numpy as np
import torch
from PIL import Image

device=torch.device("cuda")
# change to GPU later if required
model = Model(device=device) 

# need to import the latent code
codes = np.load("/home/sai_k/DualStyleGAN/instyle_2.npy", allow_pickle=True)
codes = torch.from_numpy(codes).to(device)

# exstyles = np.load("/home/sai_k/DualStyleGAN/checkpoint/cartoon/refined_exstyle_code.npy", allow_pickle='TRUE').item()
# style_id = 84
# stylename = list(exstyles.keys())[style_id]
# codes = torch.tensor(exstyles[stylename]).to(device)

# this bit is the decoder
# NOTE: both of the returned items are pytorch tensors
# NOTE: ignore the result_latent code -> it's utter garbage, just encodes a different face completely.
img_rec, result_latent = model.encoder.decoder(
	[codes],
    input_is_latent=False,
    randomize_noise=False,
    return_latents=True,
    z_plus_latent=True
)

img_rec = torch.clamp(img_rec.detach(), -1, 1)
img_rec = model.postprocess(img_rec[0])
# np.save("./new_result_latent_2", result_latent.detach().numpy())

# arr = np.load("/home/sai_k/DualStyleGAN/gui/reconstructed_face.npy", allow_pickle=True)

# Converting the numpy array into image
img  = Image.fromarray(img_rec)
# Saving the image
img.save("instyle_2_face.jpg")
print(" The Image is saved successfully")


# python encoder/InterFaceGAN/generate_data.py --model_name stylegan_ffhq --output_dir /home/sai_k/DualStyleGAN/pics/random_face_stylegan1 --latent_space_type wp --latent_codes_path /home/sai_k/DualStyleGAN/gui/latent_codes/woman.npy