"""
This file will allow the user to modify the generated image by performing some basic edits: smile, pose
Use the facial_editing directory to do the work.
"""
import numpy as np
from torch import from_numpy, clamp

HAIR_COLOUR_BOUNDARY = "../hyperplanes/boundary_Black_Hair.npy" 
GENDER_BOUNDARY_PATH_2 = "../hyperplanes/boundary_Gender.npy"
AGE_BOUNDARY = "../hyperplanes/boundary_Pale_Skin.npy" 
BALDNESS_path = "../hyperplanes/boundary_Bald.npy"

class FaceModifier2: 
    def __init__(self, model, device):
        self.model = model
        self.device = device
        # remember to unsqueeze them before adding to the latent code
        self.age_boundary = np.expand_dims(np.load(HAIR_COLOUR_BOUNDARY,allow_pickle=True), axis=0)
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
