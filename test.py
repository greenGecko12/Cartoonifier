import numpy as np

# # path = "/dcs/20/u2004267/CS310/DualStyleGAN/facial_editing/boundaries/stylegan_celebahq_pose_boundary.npy"
# path = "/home/sai_k/DualStyleGAN/facial_editing/boundaries/stylegan_celebahq_age_w_boundary.npy"
# boundary = None


from PIL import Image
import torchvision.transforms as transforms
from util import save_image, load_image, load_cartoon_image, load_viz_image, get_default_device

# boundary = np.load(path)

# if boundary is None:
#     print("Boundary has not been loaded in yet")
# else:
#     print(boundary.shape)
#     # v = boundary.copy()
#     # v = v/np.linalg.norm(v)
#     # print(v[0][1::20])
#     # print(np.linalg.norm(v))

import torch
import os
from argparse import Namespace
device = get_default_device()
from model.encoder.psp import pSp 
from torch.nn import functional as F
import torchvision
model_path = os.path.join("./checkpoint/", 'encoder.pt')
path = "/home/sai_k/DualStyleGAN/pics/000000.jpg"

ckpt = torch.load(model_path, map_location='cpu') # checkpoint
opts = ckpt['opts']
opts['checkpoint_path'] = model_path
opts = Namespace(**opts) # passing in keyword arguments
opts.device = device
encoder = pSp(opts) # (Pixel2style2pixel) pSp is the thing that generates the intermediate image
encoder.eval()
encoder.to(device)

I = load_image(path).to(device)


img_rec, instyle = encoder(F.adaptive_avg_pool2d(I, 256), randomize_noise=False, return_latents=True, 
                        z_plus_latent=True, return_z_plus_latent=True, resize=False) 

img_rec = torch.clamp(img_rec.detach(), -1, 1) 
viz=[I, img_rec]
# print(type(I), type(img_rec), type(instyle))

np.save("./instyle_2", instyle.detach().cpu().numpy())

# save_image(I.cpu(), "original_face.jpg")
# save_image(img_rec.cpu(), "reconstructed_face.jpg")
save_image(torchvision.utils.make_grid(F.adaptive_avg_pool2d(torch.cat(viz, dim=0), 256), 5, 2).cpu(), 
            "./output.jpg")
print("finished")


