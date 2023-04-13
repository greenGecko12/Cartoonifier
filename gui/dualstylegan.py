from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
from tempfile import TemporaryFile
import sys
from typing import Callable

import dlib
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from time import sleep

sys.path.append("..")

if os.getenv('SYSTEM') == 'spaces':
    os.system("sed -i '10,17d' DualStyleGAN/model/stylegan/op/fused_act.py")
    os.system("sed -i '10,17d' DualStyleGAN/model/stylegan/op/upfirdn2d.py")

app_dir = pathlib.Path(__file__).parent
submodule_dir = app_dir / 'DualStyleGAN'
sys.path.insert(0, submodule_dir.as_posix())

from model.dualstylegan import DualStyleGAN
from model.encoder.align_all_parallel import align_face
from model.encoder.psp import pSp

class Model:
    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)
        self.landmark_model = self._create_dlib_landmark_model()
        self.encoder = self._load_encoder()
        self.transform = self._create_transform()

        self.style_types = ['cartoon']
        self.generator_dict = {
            style_type: self._load_generator(style_type)
            for style_type in self.style_types
        }
        self.exstyle_dict = {
            style_type: self._load_exstylecode(style_type)
            for style_type in self.style_types
        }

    @staticmethod
    def _create_dlib_landmark_model():
        url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
        path = pathlib.Path('../checkpoint/shape_predictor_68_face_landmarks.dat')
        # path = 'checkpoint/shape_predictor_68_face_landmarks.dat'
        # file = os.file.open()
        if not path.exists():
            # bz2_path = 'shape_predictor_68_face_landmarks.dat.bz2'
            # torch.hub.download_url_to_file(url, bz2_path)
            # subprocess.run(f'bunzip2 -d {bz2_path}'.split())
            print("path not found")
        return dlib.shape_predictor(path.as_posix())

    def _load_encoder(self) -> nn.Module:
        # ckpt_path = huggingface_hub.hf_hub_download(MODEL_REPO, 'models/encoder.pt')
        ckpt_path = '../checkpoint/encoder.pt'
        ckpt = torch.load(ckpt_path, map_location='cpu')
        opts = ckpt['opts']
        opts['device'] = self.device.type
        opts['checkpoint_path'] = ckpt_path
        opts = argparse.Namespace(**opts)
        model = pSp(opts)
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _create_transform() -> Callable:
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(256),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        return transform

    def _load_generator(self, style_type: str) -> nn.Module:
        model = DualStyleGAN(1024, 512, 8, 2, res_index=6)
        ckpt_path = '../checkpoint/cartoon/generator.pt'
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['g_ema'])
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _load_exstylecode(style_type: str) -> dict[str, np.ndarray]:
        path = '../checkpoint/cartoon/refined_exstyle_code.npy'
        exstyles = np.load(path, allow_pickle=True).item()
        return exstyles

    def detect_and_align_face(self, image) -> np.ndarray:
        image = align_face(filepath=image, predictor=self.landmark_model)
        return image

    @staticmethod
    def denormalize(tensor: torch.Tensor) -> torch.Tensor:
        return torch.clamp((tensor + 1) / 2 * 255, 0, 255).to(torch.uint8)

    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        tensor = self.denormalize(tensor)
        return tensor.cpu().numpy().transpose(1, 2, 0)

    # @torch.inference_mode()
    def reconstruct_face(self, image: np.ndarray, PIL_true:bool=True) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
        if PIL_true:
            image = PIL.Image.fromarray(image)
        input_data = self.transform(image).unsqueeze(0).to(self.device)
        img_rec, instyle = self.encoder(input_data,
                                        randomize_noise=False,
                                        return_latents=True,
                                        z_plus_latent=True,
                                        return_z_plus_latent=True,
                                        resize=False)
        img_rec = torch.clamp(img_rec.detach(), -1, 1)
        img_rec = self.postprocess(img_rec[0])
        # np.save("./randomface",instyle.detach().numpy())
        return img_rec, instyle, img_rec, instyle
    
    def encode_cartoon_face(self, image , skipAlignment: bool): # image for now is the PATH
        if skipAlignment: # DOESN'T WORK
            empty = np.zeros((1024, 1024, 3))
            image_2 = dlib.load_rgb_image(image)
            # print(type(image_2))
            # image_2 = np.load(image, allow_pickle=True)
            # sleep(2)
            # calling the method just above to reconstuct the cartoon face in StyleGAN's latent space
            
            # img_rec, exstyle, _ = self.reconstruct_face(image) 
            # img_rec = torch.clamp(img_rec.detach(), -1, 1)
            # img_rec = self.postprocess(img_rec[0])          

            img_rec, exstyle = self._helper(image_2, True)
            return empty, img_rec, exstyle
        else: # when facial alignment has been ENABLED ##################### WORKS
            # image_path = np.save("/home/sai_k/DualStyleGAN/gui/latent_codes/cartoon", image)

            # outfile = TemporaryFile()
            # np.save(outfile, image)
            aligned_face = self.detect_and_align_face(image) # returns PIL image

            # img_rec, exstyle, _ = self.reconstruct_face(aligned_face) 
            # img_rec = torch.clamp(img_rec.detach(), -1, 1)
            # img_rec = self.postprocess(img_rec[0])

            """The line below might come in handy if error about CPU/GPU"""
            # instyle = instyle.detach().cpu().numpy()
            img_rec, exstyle = self._helper(aligned_face, False)
            return aligned_face, img_rec, exstyle

    def _helper(self, image, flag):
        img_rec, exstyle, x, y = self.reconstruct_face(image, flag) 
        # variables x & y are simply not used
        # img_rec = torch.clamp(img_rec.detach(), -1, 1)
        # img_rec = self.postprocess(img_rec[0])

        return img_rec, exstyle

    # @torch.inference_mode()
    # Copy the modified code that accepts two cartoon styles
    def generate(self, style_type: str, style_id: int, structure_weight: float,
                 color_weight: float, structure_only: bool,
                 instyle: torch.Tensor, style_id_1: int, weight:int, weight_1:int, user_exstyle) -> np.ndarray:
        generator = self.generator_dict[style_type]

        if user_exstyle is None:
            exstyles = self.exstyle_dict[style_type]
            all_stylenames = list(exstyles.keys())

            style_id  = int(style_id)
            stylename = all_stylenames[style_id]
            latent = torch.tensor(exstyles[stylename]).to(self.device) 
            
            style_id_1 = int(style_id_1)
            if style_id_1 != -1:
                stylename_1 = all_stylenames[style_id_1]
                latent_1 = torch.tensor(exstyles[stylename_1]).to(self.device)
                latent = latent*weight + latent_1*weight_1
        else:
            # print(type(user_exstyle))
            # if user provides their own cartoon image, then that takes more priority
            latent = user_exstyle.to(self.device)

        # copy the bit of the style_transfer.py file that corresponds to 2 images
        if structure_only:
            latent[0, 7:18] = instyle[0, 7:18]
        exstyle = generator.generator.style(
            latent.reshape(latent.shape[0] * latent.shape[1],
                           latent.shape[2])).reshape(latent.shape)

        img_gen, _ = generator([instyle],
                               exstyle,
                               z_plus_latent=True,
                               truncation=0.7,
                               truncation_latent=0,
                               use_res=True,
                               interp_weights=[structure_weight] * 7 +
                               [color_weight] * 11)
        img_gen = torch.clamp(img_gen.detach(), -1, 1)
        img_gen = self.postprocess(img_gen[0])
        return img_gen