from dualstylegan import Model
import numpy as np
import torch
from PIL import Image

device=torch.device("cpu")
# change to GPU later if required
model = Model(device=device) 

# need to import the latent code
codes = np.load("/home/sai_k/DualStyleGAN/gui/new_result_latent.npy", allow_pickle=True)
codes = torch.from_numpy(codes)

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


np.save("./new_result_latent_2", result_latent.detach().numpy())

# arr = np.load("/home/sai_k/DualStyleGAN/gui/reconstructed_face.npy", allow_pickle=True)

# Converting the numpy array into image
img  = Image.fromarray(img_rec)
# Saving the image
img.save("/home/sai_k/DualStyleGAN/gui/new_photo_2.jpg")
print(" The Image is saved successfully")

