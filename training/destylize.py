"""
This file basically does what is mentioned in section 3.1 of the paper: Facial Destylization 

Stage 1: Latent initialization
Stage 2: Latent optimization
Stage 3: Image embedding

In the paper, it mentions something about facial destylization -> transforming artistic portraits into realistic faces.
This apparently provides face-portrait pairs which act as supervision during Progressive fine-tuning.

Don't think this file is that important for implementing my extensions.
"""
import os
import numpy as np
import torch
from torch import optim
from util import save_image
import argparse
from argparse import Namespace
from torchvision import transforms
from torch.nn import functional as F
import torchvision
from PIL import Image
from tqdm import tqdm # progress bars
import math

from model.stylegan.model import Generator
from model.stylegan import lpips
from model.encoder.psp import pSp
from model.encoder.criteria import id_loss

class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Facial Destylization")
        self.parser.add_argument("style", type=str, help="target style type")
        self.parser.add_argument("--truncation", type=float, default=0.7, help="truncation for intrinsic style code (content)")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path of the saved models")
        self.parser.add_argument("--model_name", type=str, default='finetune-000600.pt', help="name of the saved fine-tuned model")
        self.parser.add_argument("--data_path", type=str, default='./data/', help="path of dataset")
        self.parser.add_argument("--iter", type=int, default=300, help="total training iterations")
        self.parser.add_argument("--batch", type=int, default=1, help="batch size")

    def parse(self):
        self.opt = self.parser.parse_args()        
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt

# something learning rate
def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

# Not entirely sure what this bit actually does
def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss

# probably just a utility method or something
def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)
        

if __name__ == "__main__":
    device = "cuda"

    parser = TestOptions()
    args = parser.parse()
    print('*'*50)
    
    # maing a directory to store all the destylized portraits
    if not os.path.exists("log/%s/destylization/"%(args.style)):
        os.makedirs("log/%s/destylization/"%(args.style))
        
    # again, just composing various transformations
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    
    # fine-tuned StyleGAN g'
    generator_prime = Generator(1024, 512, 8, 2).to(device)
    generator_prime.eval()
    # orginal StyleGAN g
    generator = Generator(1024, 512, 8, 2).to(device)
    generator.eval()

    # I guess this checkpoint thing loads in a dictionary with the weights of all the models etc.
    ckpt = torch.load(os.path.join(args.model_path, args.style, args.model_name))
    # loading the weights of the fine-tuned model
    generator_prime.load_state_dict(ckpt["g_ema"])

    # loading the the original StyleGAN g that was trained on FFHQ faces
    ckpt = torch.load(os.path.join(args.model_path, 'stylegan2-ffhq-config-f.pt'))
    generator.load_state_dict(ckpt["g_ema"])

    # loading and instantiating various objects
    noises_single = generator.make_noise()
    model_path = os.path.join(args.model_path, 'encoder.pt')
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts)
    encoder = pSp(opts)
    encoder.eval() # evaluation mode
    encoder.to(device)

    # variables (objects) to store the losses of the models during training --> used to update the weights
    percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=device.startswith("cuda"))
    id_loss = id_loss.IDLoss(os.path.join(args.model_path, 'model_ir_se50.pth')).to(device).eval()

    print('Load models successfully!')
    
    datapath = os.path.join(args.data_path, args.style, 'images/train')
    files = os.listdir(datapath) 
    
    dict = {}
    dict2 = {}
    for ii in range(0,len(files),args.batch):
        batchfiles = files[ii:ii+args.batch]
        imgs = []
        for file in batchfiles:
            img = transform(Image.open(os.path.join(datapath, file)).convert("RGB"))
            imgs.append(img)
        imgs = torch.stack(imgs, 0).to(device)

        with torch.no_grad():  
            # reconstructed face g(z^+_e) and extrinsic style code z^+_e
            img_rec, latent_e = encoder(imgs, randomize_noise=False, return_latents=True, z_plus_latent=True) # using pSp to embed the portrait into StyleGAN latent space
            
        for j in range(imgs.shape[0]):
            dict2[batchfiles[j]] = latent_e[j:j+1].cpu().numpy()
        
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())
        for noise in noises:
            noise.requires_grad = True    

        # z^+ to be optimized in Eq. (1)
        latent = latent_e.detach().clone()
        latent.requires_grad = True

        optimizer = optim.Adam([latent] + noises, lr=0.1)

        pbar = tqdm(range(args.iter))

        for i in pbar:
            t = i / args.iter
            lr = get_lr(t, 0.1)
            optimizer.param_groups[0]["lr"] = lr
            latent_n = latent

            # g'(z^+)
            img_gen, _ = generator_prime([latent_n], input_is_latent=False, noise=noises, z_plus_latent=True)

            batch, channel, height, width = img_gen.shape

            if height > 256:
                factor = height // 256

                img_gen = img_gen.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                img_gen = img_gen.mean([3, 5])

            Lperc = percept(img_gen, imgs).sum()
            LID = id_loss(img_gen, imgs)
            Lreg = latent.std(dim=1).mean()
            Lnoise = noise_regularize(noises)

            loss = Lperc + 0.1 * LID + Lreg + 1e5 * Lnoise

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise_normalize_(noises)

            pbar.set_description(
                (
                    f"[{ii:03d}/{len(files):03d}]"
                    f" Lperc: {Lperc.item():.3f}; Lnoise: {Lnoise.item():.3f};"
                    f" LID: {LID.item():.3f}; Lreg: {Lreg.item():.3f}; lr: {lr:.3f}"
                )
            )

        with torch.no_grad():
            # (optional) preserve color
            latent[:,8:18] = latent_e[:,8:18].detach()   
            # g(hat(z)^+_e)
            img_dsty, _ = generator([latent.detach()], input_is_latent=False, truncation=args.truncation, 
                           truncation_latent=0, noise=noises, z_plus_latent=True)
            img_dsty = F.adaptive_avg_pool2d(img_dsty.detach(), 256)
            # (optional) preserve color
            _, latent_i = encoder(img_dsty, randomize_noise=False, return_latents=True, z_plus_latent=True)
            # z^+_i
            latent_i[:,8:18] = latent_e[:,8:18].detach()   
            # g(hat(z)^+_i)
            img_refine, _ = generator([latent_i.detach()], input_is_latent=False, truncation=args.truncation, 
                           truncation_latent=0, noise=noises, z_plus_latent=True)
            img_refine = F.adaptive_avg_pool2d(img_refine.detach(), (256,256))

            for j in range(imgs.shape[0]):
                vis = torchvision.utils.make_grid(torch.cat([imgs[j:j+1], img_rec[j:j+1].detach(), 
                                                             img_dsty[j:j+1].detach(), img_refine[j:j+1].detach()], dim=0), 4, 1)
                save_image(torch.clamp(vis.cpu(),-1,1), os.path.join("./log/%s/destylization/"%(args.style), batchfiles[j]))
                dict[batchfiles[j]] = latent_i[j:j+1].cpu().numpy()
    
    np.save(os.path.join(args.model_path, args.style, 'instyle_code.npy'), dict)    
    np.save(os.path.join(args.model_path, args.style, 'exstyle_code.npy'), dict2) 
    print('Destylization done!')
    
