"""
Contains the python code to find the face in the inputted image, crop and align it
"""

# def reconstruct_face(self,
#                         image: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
#     image = PIL.Image.fromarray(image)
#     input_data = self.transform(image).unsqueeze(0).to(self.device)
#     img_rec, instyle = self.encoder(input_data,
#                                     randomize_noise=False,
#                                     return_latents=True,
#                                     z_plus_latent=True,
#                                     return_z_plus_latent=True,
#                                     resize=False)
#     img_rec = torch.clamp(img_rec.detach(), -1, 1)
#     img_rec = self.postprocess(img_rec[0])
#     np.save("./instyle",instyle.detach().numpy())
#     return img_rec, instyle, img_rec