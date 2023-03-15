"""
This file will allow the user to modify the generated image by performing some basic edits: smile, pose
Use the facial_editing directory to do the work.
"""
import numpy as np
from torch import from_numpy, clamp

HAIR_COLOUR_BOUNDARY = "/home/sai_k/DualStyleGAN/hyperplanes/stylegan2_ffhq_z/boundary_Black_Hair.npy" # Works really well
GENDER_BOUNDARY_PATH_2 = "/home/sai_k/DualStyleGAN/hyperplanes/stylegan2_ffhq_z/boundary_Gender.npy" # WORKS really well
# HAIR_COLOUR_BOUNDARY = "/home/sai_k/DualStyleGAN/hyperplanes/age.npy"

AGE_BOUNDARY = "/home/sai_k/DualStyleGAN/hyperplanes/stylegan2_ffhq_z/boundary_Pale_Skin.npy" # leave this as is

BALDNESS_path = "/home/sai_k/DualStyleGAN/hyperplanes/stylegan2_ffhq_z/boundary_Bald.npy" # Works well

"""

(1 1 512) 
(1 18, 512)
Notes for each hyperplane

**Age: DOES NOT WORK AT ALL
**Gender: Works
**Smile: DOES NOT WORK AT ALL
**Yaw: DOES NOT WORK WELL

*Eye Ratio: DOES NOT WORK AT ALL
*Lip Ratio: Works as smile
*Mouth Open: Works as Smile quite well
*Roll: not well
"""

class FaceModifier2: 
    def __init__(self, model, device):
        self.model = model # dualstylegan.py
        self.device = device
        # remember to unsqueeze them before adding to the latent code
        self.age_boundary = np.expand_dims(np.load(HAIR_COLOUR_BOUNDARY,allow_pickle=True), axis=0)
        # self.age_boundary = np.expand_dims(np.repeat(np.load(HAIR_COLOUR_BOUNDARY,allow_pickle=True),18,axis=0), axis=0)
        self.pose_boundary = np.expand_dims(np.load(AGE_BOUNDARY,allow_pickle=True), axis=0)
        self.gender_boundary = np.expand_dims(np.load(GENDER_BOUNDARY_PATH_2,allow_pickle=True), axis=0)
        self.smile_boundary = np.expand_dims(np.load(BALDNESS_path,allow_pickle=True), axis=0)
        self.latent_code = np.empty((1,18,512))
        self.latent_code_copy = np.empty((1,18,512))

    def set_latent_code(self, latent_code):
        self.latent_code = latent_code
        self.latent_code_copy = latent_code

    # all the methods below conduct inferencing   
    def modify_latent_code(self, boundary, offset):
        # adding a multiple of the hyperplane
        boundary2 = boundary * offset # element-wise multiplication
        self.latent_code = self.latent_code_copy
        self.latent_code += boundary2


        
        # both of these were of shape (1, 18, 512)
        # print(boundary2.shape) 
        # print(self.latent_code.shape)

        # np.save("./modified_latent", self.latent_code)

        # code=linear_interpolate(self.latent_code,boundary,start_distance=offset,end_distance=offset,steps=1)

        # print(code.shape)
        # print("============================================================")
        # print("modified latent code is of shape:")
        # print(code.shape)
        # print("============================================================")

        # modified_code = np.add(self.latent_code, np.multiply(boundary, offset))
        # return modified_code
        # self.latent_code
    
    # each method only returns 1 image  --> offset = start_distance = end_distance
    def change_age(self, offset):
        self.modify_latent_code(self.age_boundary, offset)

    def change_pose(self, offset):
        self.modify_latent_code(self.pose_boundary, offset)
    
    def change_smile(self, offset):
        self.modify_latent_code(self.smile_boundary, offset)

    def change_gender(self, offset):
        self.modify_latent_code(self.gender_boundary, offset)
    
    def modify(self, age_offset,  gender_offset, pose_offset, smile_offset):
        # preprocessing
        # self.latent_code = self.model.preprocess(self.latent_code, **self.kwargs)
        
        # no modifications made to the image
        if (age_offset == 0) and (pose_offset == 0) and (smile_offset == 0) and (gender_offset == 0):
            return self.synthesize(), self.latent_code

        if age_offset != 0:
            self.change_age(age_offset)
        if pose_offset != 0:
            self.change_pose(pose_offset)
        if smile_offset != 0:
            self.change_pose(smile_offset)
        if gender_offset != 0:
            self.change_gender(gender_offset)

        output = self.synthesize()
        return output, from_numpy(self.latent_code).to(self.device)

    def synthesize(self):
        codes2 = from_numpy(self.latent_code).to(self.device)
        # this bit is the decoder
        # NOTE: both of the returned items are pytorch tensors
        # NOTE: ignore the result_latent code -> it's utter garbage, just encodes a different face completely.
        img_rec, _ = self.model.encoder.decoder(
            [codes2],
            input_is_latent=False,
            randomize_noise=False,
            return_latents=True,
            z_plus_latent=True
        )

        img_rec = clamp(img_rec.detach(), -1, 1)
        img_rec = self.model.postprocess(img_rec[0])
        return img_rec


# need to import the latent code
# codes = np.load("/home/sai_k/DualStyleGAN/gui/randomface.npy", allow_pickle=True)
# boundary = np.load("/home/sai_k/DualStyleGAN/stylegan2directions/gender.npy", allow_pickle=True)
# boundary1 = np.expand_dims(boundary, axis=0)

# boundary2 = np.multiply(boundary1, -8)


# codes -= boundary2
# codes2 = from_numpy(codes)

# print("The dimensions of the generated image", end=":")
# print(img_rec.shape)

# Converting the numpy array into image
# img  = Image.fromarray(img_rec)
# print(type(img))
# Saving the image
# img.save(f"/home/sai_k/DualStyleGAN/gui/gender/modified_man_3.jpg")

# print("Images saved successfully")

# from dualstylegan import Model
# from PIL import Image

# NOTE: we don't save the result_latent code cos it's utter garbage, just encodes a different face completely.
# np.save("./new_result_latent_2", result_latent.detach().numpy())
# arr = np.load("/home/sai_k/DualStyleGAN/gui/reconstructed_face.npy", allow_pickle=True)