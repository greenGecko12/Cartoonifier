"""
In this file, have a quick demo to demonstrate the effects of modifying the latent code on the output image

Average face with all zeros: demonstrate how that looks like
Random face - have a button to create a random face
Display the latent code for that face 
    --> in this case just the z code 
    --> find a easy to understand way for the user to manipulate the z vector
    --> feed this code back into the generator and show the resulting face

Have another example using a GAN to generate cars, maybe something else as well.


MAYBE INCLUDE THIS WHOLE UI TO THE BOTTOM OF APP.PY ==> BELOW STEP 5 (Generate cartoon character step)
"""

import gradio as gr
