# 3rd year dissertation project code 

## Please note that this repository will ONLY work on a Linux machine with an NVIDIA GPU. 

Note: The process to install all the required dependencies and PyTorch models will take around an **hour**.
<br>

**Dependencies:**

All dependencies for defining the environment are provided in `environment/dualstylegan_env.yaml`.
It is recommended to runthis repository using [Anaconda](https://docs.anaconda.com/anaconda/install/):
```bash
conda env create -f ./environment/dualstylegan_env.yaml
```
I use CUDA 10.1 so it will install PyTorch 1.7.1 (corresponding to [Line 22](https://github.com/williamyang1991/DualStyleGAN/blob/main/environment/dualstylegan_env.yaml#L22), [Line 25](https://github.com/williamyang1991/DualStyleGAN/blob/main/environment/dualstylegan_env.yaml#L25), [Line 26](https://github.com/williamyang1991/DualStyleGAN/blob/main/environment/dualstylegan_env.yaml#L26) of `dualstylegan_env.yaml`). Please install PyTorch that matches your own CUDA version following [https://pytorch.org/](https://pytorch.org/).
Make sure you are in your virtual environement; this can be done by the command 
```bash
conda activate dualstyle_env_yaml
```

## Inference for Style Transfer and Artistic Portrait Generation
### Pretrained Models

Pretrained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1GZQ6Gs5AzJq9lUL-ldIQexi0JYPKNy8b?usp=sharing) (access code: cvpr):

| Model | Description |
| :--- | :--- |
| [encoder](https://drive.google.com/file/d/1NgI4mPkboYvYw3MWcdUaQhkr0OWgs9ej/view?usp=sharing) | Pixel2style2pixel encoder that embeds FFHQ images into StyleGAN2 W+ latent code |
| [cartoon](https://drive.google.com/drive/folders/1xPo8PcbMXzcUyvwe5liJrfbA5yx4OF1j?usp=sharing) | DualStyleGAN and sampling models trained on Cartoon dataset, 317 (refined) extrinsic style codes |

The saved checkpoints are under the following folder structure:
```
checkpoint
|--encoder.pt                     % Pixel2style2pixel model
|--cartoon
    |--generator.pt               % DualStyleGAN model
    |--sampler.pt                 % The extrinsic style code sampling model
    |--exstyle_code.npy           % extrinsic style codes of Cartoon dataset
    |--refined_exstyle_code.npy   % refined extrinsic style codes of Cartoon dataset
```
## Using the GUI - recommended
If you want to use the GUI, then please do:
```
pip install gradio
```
Gradio is the UI library that allows web-based UI to be built. 
Then navigate to the "gui" directory and enter:
```
python app.py
```
The GUI is much more intuitive to use than through the terminal. If you simply want to use the GUI then you don't need to read the instructions below.

Click the link given to you on the terminal by Gradio. If nothing opens, open your browser to port 7860 on localhost.<br>
```http://127.0.0.1:7860```

If there are any errors, this is most likely due to incorrect paths, please change the absolute file paths as per your system.

<hr>
<hr>

## Exemplar-Based Style Transfer - not recommended
Transfer the style of a default Cartoon image onto a default face:
```python
python style_transfer.py 
```
Specify the style image with `--style` and `--style_id` find the visual mapping between id and the style image [here](./doc_images/cartoon_overview.jpg)). Specify the filename of the saved images with `--name`. Specify the weight to adjust the degree of style with `--weight`.
```python
python style_transfer.py
python style_transfer.py --style cartoon --style_id 10
```

Specify the content image with `--content`. If the content image is not well aligned with FFHQ, use `--align_face`. For preserving the color style of the content image, use `--preserve_color` or set the last 11 elements of `--weight` to all zeros.
```python
python style_transfer.py --content ./data/content/unsplash-rDEOVtE7vOs.jpg --align_face --preserve_color \
       --style arcane --name arcane_transfer --style_id 13 \
       --weight 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 1 1 1 1 1 1 1 
```

More options can be found via `python style_transfer.py  -h`.

## Acknowledgments

The code is mainly developed based on [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch), [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel) and [PasticheMaster](https://github.com/williamyang1991/DualStyleGAN)