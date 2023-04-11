""" 
Encodes a cartoon face into stylegan's latent space
"""
import numpy as np
import torch
from util import load_image, get_default_device
from argparse import Namespace
from torchvision import transforms
from torch.nn import functional as F
from model.encoder.psp import pSp 
from PIL import Image

# entry point to the code
if __name__ == "__main__":
    device = get_default_device()
    
    # transforming the data before putting it through the model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])
    
    model_path = "/home/sai_k/DualStyleGAN/checkpoint/encoder.pt"
    content = "/home/sai_k/DualStyleGAN/data/cartoon/images/train/Cartoons_00558_01.jpg"

    ckpt = torch.load(model_path, map_location='cpu') # checkpoint
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts) # passing in keyword arguments
    opts.device = device
    encoder = pSp(opts) # (Pixel2style2pixel) pSp is the thing that generates the intermediate image
    encoder.eval()
    encoder.to(device)

    # disable gradient calculation
    with torch.no_grad():
        I = load_image(content).to(device) # converts to Tensor, normalises and then moves to the GPU
        # 256 is the output dimension
        img_rec, instyle = self.encoder(F.adaptive_avg_pool2d(I, 256), randomize_noise=False, return_latents=True, 
                                z_plus_latent=True, return_z_plus_latent=True, resize=False)   

        instyle = instyle.detach().cpu().numpy()

        np.save("./anna_2", instyle)
        img_rec = torch.clamp(img_rec.detach(), -1, 1).detach().cpu().numpy()


        img  = Image.fromarray(img_rec)
        # Saving the image
        img.save("/home/sai_k/DualStyleGAN/gui/anna.jpg")
        print(" The Image is saved successfully")
                
        # return instyle, img_rec

