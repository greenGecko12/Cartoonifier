from dualstylegan import Model
import numpy as np
import torch
from PIL import Image

device="cpu"
# change to GPU later if required
model = Model(device=device) 

# need to import the latent code
codes = np.load("/home/sai_k/DualStyleGAN/gui/randomface.npy", allow_pickle=True)
boundary = np.load("/home/sai_k/DualStyleGAN/hyperplanes/gender.npy", allow_pickle=True)
boundary1 = np.expand_dims(boundary, axis=0)

# print("Shape of boundary", end=": ")
# print(boundary.shape)

# print("Shape of new boundary is", end=": ")
# print(boundary2.shape)

# for i in range(1, 10):

boundary2 = np.multiply(boundary1, 5)

# NOTE: np.add() causes this error: RuntimeError: expected scalar type Double but found Float
# NOTE: just use '+=' for now 
# codes = np.add(codes, boundary2) 

codes += boundary2

codes2 = torch.from_numpy(codes)

# this bit is the decoder
# NOTE: both of the returned items are pytorch tensors
# NOTE: ignore the result_latent code -> it's utter garbage, just encodes a different face completely.
img_rec, result_latent = model.encoder.decoder(
    [codes2],
    input_is_latent=False,
    randomize_noise=False,
    return_latents=True,
    z_plus_latent=True
)

img_rec = torch.clamp(img_rec.detach(), -1, 1)
img_rec = model.postprocess(img_rec[0])

# print("The dimensions of the generated image", end=":")
# print(img_rec.shape)

# Converting the numpy array into image
img  = Image.fromarray(img_rec)
# print(type(img))
# Saving the image
img.save(f"/home/sai_k/DualStyleGAN/gui/gender/modified_man_7.jpg")

print("Images saved successfully")


# NOTE: we don't save the result_latent code cos it's utter garbage, just encodes a different face completely.
# np.save("./new_result_latent_2", result_latent.detach().numpy())
# arr = np.load("/home/sai_k/DualStyleGAN/gui/reconstructed_face.npy", allow_pickle=True)


