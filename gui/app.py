"""
This file will add the Gradio GUI to make the GAN much easier to use. 
"""
from __future__ import annotations

import argparse
import pathlib

import gradio as gr

# importing the Generator (a brief version of style_transfer.py)
from dualstylegan import Model

# import the main() function from the edit.py file in the facial_editing directory
DESCRIPTION = '<h1>Cartoonify your face with the power of GANs!</h1>'

SECTION = '''## Step 1: Upload your image
- Please upload an image containing a near-frontal face to the **Input Image**.
    - If there are multiple faces in the image, hit the Edit button in the upper right corner and crop the input image beforehand.
- Click the **Preprocess** button.
    - The final result will be based on this **Reconstructed Face**. So, if the reconstructed image is not satisfactory, you may want to change the input image.
'''

# returns the URL from which the cartoon grid is fetched from
def get_style_image_markdown_text():
    url = 'https://raw.githubusercontent.com/williamyang1991/DualStyleGAN/main/doc_images/cartoon_overview.jpg'
    return f'<img id="style-image" src="{url}" alt="cartoon style images">'


def main():
    device="cpu"
    model = Model(device=device) # change to GPU later if required

    def show_styles():
        style_index_1.value
        style_index_2.value

    with gr.Blocks(css='style.css') as demo:
        gr.HTML(DESCRIPTION)

        with gr.Box():
            gr.Markdown(SECTION)
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        input_image = gr.Image(label='Input Image',
                                               type='filepath')
                    with gr.Row():
                        preprocess_button = gr.Button('Preprocess')
                with gr.Column():
                    with gr.Row():
                        aligned_face = gr.Image(label='Aligned Face',
                                                type='numpy',
                                                interactive=False)
                with gr.Column():
                    reconstructed_face = gr.Image(label='Reconstructed Face',
                                                  type='numpy')
                    instyle = gr.Variable()

        with gr.Box():
            gr.Markdown('''## Step 2: Pick cartoon style
                        - If you pick two cartoon styles then the output will have characteristics from both cartoon images.
                            - You will also be able to select the proportion of each style that you would like the 
                            output from the generator to have in the next step.
                        ''')
            with gr.Row():
                with gr.Column():
                    # fetching the image with all the cartoon characters and numbers
                    text = get_style_image_markdown_text()
                    style_image = gr.Markdown(value=text)

                    style_index_1 = gr.Slider(0,316,value=26,step=1,label='Style Image Index 1', interactive=True)
                    style_index_2 = gr.Slider(-1,316,value=-1,step=1,label='Style Image Index 2', interactive=True)
                    confirm_styles = gr.Button("Confirm choices")
            
            gr.HTML('''<p></p><p></p>''') # adding some space with some block-level tags
            gr.Markdown("Picking the second cartoon style is **OPTIONAL**. Please leave it at -1 if you don't want to select a second style.")

        with gr.Box():
            gr.Markdown('''## Step 3: Mixing the two styles
                        - In case you couldn't quite see the cartoon face(s), here is what you have selected.
                        - Move the slider to suit your style preference.
                            - Moving the slider to the left will make the output look more like Style 1.
                            - A central position the default position) will incorporate the styles of both cartoon characters with equal weight.
                            - Likewise, moving the slider to the right will make the output look more like Style 2.
                        ''')
            with gr.Row():
                with gr.Column():
                    cartoon_style_1 = gr.Image(label='cartoon_style_1',type='numpy',
                                                interactive=False)
                with gr.Column():
                    cartoon_style_2 = gr.Image(label='cartoon_style_2', type='numpy',
                                                interactive=False)
                    
            gr.HTML('''<p></p><p></p>''')
            weights = gr.Slider(0, 100, 50, step=0.10, label='Specify combination', interactive=True, show_label=False)

        with gr.Box():
            gr.Markdown('''## Step 4: (**OPTIONAL**) Facial Modification
                        - Change several attributes of the input face.
                            - These will be reflected in the generated cartoon face.
                        - On the left is the reconstructed face you saw in step 1 for your reference.
                            - Any modifications will be applied to this face to create what's on the right.
                        ''')
            with gr.Row():
                with gr.Column():
                    original = gr.Image(label='Reconstructed Face', type='numpy', interactive=False)
                
                with gr.Column():
                    age = gr.Slider(-1, 1, 0, step=0.1, label='Age')
                    # gender = gr.Slider(-1, 1, 0, step=0.1, label='Gender')
                    pose = gr.Slider(-1, 1, 0, step=0.1, label='Pose')
                    smile = gr.Slider(-1, 1, 0, step=0.1, label='Smile')
                    confirm_modified_face = gr.Button("Modify my face!")

                with gr.Column():
                    modified = gr.Image(label='Modified Image', type='numpy', interactive=False)

        with gr.Box():
            gr.Markdown('''## Step 5: Create cartoon character
                        - Adjust **Structure Weight** and **Color Weight**.
                            - These are weights for the style image, so the larger the value, the closer the resulting image will be to the style image.
                        - Hit the **Generate** button.
                        ''')
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        structure_weight = gr.Slider(0,1,value=0.6,step=0.1,label='Structure Weight')
                    with gr.Row():
                        color_weight = gr.Slider(0,1,value=1,step=0.1,label='Color Weight')
                    with gr.Row():
                        structure_only = gr.Checkbox(label='Structure Only')
                    with gr.Row():
                        generate_button = gr.Button('Generate')

                with gr.Column():
                    result = gr.Image(label='Result')

        preprocess_button.click(
                                fn=model.detect_and_align_face,
                                inputs=input_image,
                                outputs=aligned_face)
        aligned_face.change(
                            fn=model.reconstruct_face,
                            inputs=aligned_face,
                            outputs=[
                                reconstructed_face,
                                instyle,
                            ])
        
        confirm_styles.click()
        # generate_button.click(
        #                     # fn=model.generate,
        #                     fn=hello,
        #                     inputs=[
        #                         "cartoon",
        #                         style_index,
        #                         structure_weight,
        #                         color_weight,
        #                         structure_only,
        #                         instyle,
        #                       ],
        #                       outputs=result)
        
    demo.launch()

if __name__ == '__main__':
    main()