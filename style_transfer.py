""" 
This code conducts the inferencing for the neural network. 

Extrinsic style code == this is the style of the image
Intrinsic style code == content of the image

Changing the extrinsic style code means the same face but changing the artistic style of the image
Changing the intrinsic style code corresponds to changing the actual face inputted


The unseen images are given priority to the cartoon numbers
"""
import os
import numpy as np
import torch
from util import save_image, load_image, load_cartoon_image, load_viz_image, get_default_device
import argparse
from argparse import Namespace
from torchvision import transforms
from torch.nn import functional as F
import torchvision
from model.dualstylegan import DualStyleGAN # the actual generator
from model.encoder.psp import pSp 
from PIL import Image

# this is the bit that does all the inferencing
class TestOptions():
    def __init__(self):

        # all the bits of info required for inferencing
        # look at the help argument to see what each one means
        self.parser = argparse.ArgumentParser(description="Exemplar-Based Style Transfer")

        # there are default parameters which are OVERWRITTEN when the user passes their own arguments
        self.parser.add_argument("--content", type=str, default='./data/content/randomface.jpg', help="path of the content image")
        self.parser.add_argument("--style", type=str, default='cartoon', help="target style type")
        self.parser.add_argument("--style_id", type=int, default=53, help="the id of the style image")
        self.parser.add_argument("--style_id_2", type=int, default=None, help="the id of the SECOND style image")
        self.parser.add_argument("--truncation", type=float, default=0.75, help="truncation for intrinsic style code (content)")
        self.parser.add_argument("--weight", type=float, nargs=18, default=[0.75]*7+[1]*11, help="weight of the extrinsic style")
        self.parser.add_argument("--name", type=str, default='cartoonify', help="filename to save the generated images")
        self.parser.add_argument("--preserve_color", action="store_true", help="preserve the color of the content image")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path of the saved models")
        self.parser.add_argument("--model_name", type=str, default='generator.pt', help="name of the saved dualstylegan")
        self.parser.add_argument("--output_path", type=str, default='./output3/', help="path of the output images")
        self.parser.add_argument("--data_path", type=str, default='./data/', help="path of dataset")
        self.parser.add_argument("--align_face", action="store_true", help="apply face alignment to the content image")
        self.parser.add_argument("--exstyle_name", type=str, default=None, help="name of the extrinsic style codes")

        # self.parser.add_argument("--unseen_image", type=bool, default=False, 
        # help="checking if the user is using their own reference style image")

        self.parser.add_argument("--cartoon_path", type=str, default="./cartoon_faces/", help="where the user-provided cartoon faces are")

        # path to the first cartoon image
        self.parser.add_argument("--cartoon_image_1", type=str, default=None, help="path to the FIRST user-provided image of a cartoon face")
        # path to the second cartoon image
        self.parser.add_argument("--cartoon_image_2", type=str, default=None, help="path to the SECOND user-provided image of a cartoon face")

    def parse(self):
        # parsing and returning all the arguments passed in by the user
        self.opt = self.parser.parse_args()

        # by default exstyle_name is None
        if self.opt.exstyle_name is None: # the extrinsic style path is the style of the image
            # .npy is a numpy array on the disk
            if os.path.exists(os.path.join(self.opt.model_path, self.opt.style, 'refined_exstyle_code.npy')):
                self.opt.exstyle_name = 'refined_exstyle_code.npy'
            else:
                self.opt.exstyle_name = 'exstyle_code.npy'     
            
        args = vars(self.opt) # converting to a dictionary
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt    


## WE ARE NOT IN THE CLASS ANYMORE - just functions!

def run_alignment(args):
    # pre-processing, just focusing on the face and then cropping everything else? TODO: doule check exactly what this function does
    import dlib
    from model.encoder.align_all_parallel import align_face
    modelname = os.path.join(args.model_path, 'shape_predictor_68_face_landmarks.dat')
    if not os.path.exists(modelname):
        print("is downloading the .dat file")
        import wget, bz2
        wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname+'.bz2')
        zipfile = bz2.BZ2File(modelname+'.bz2')
        data = zipfile.read()
        open(modelname, 'wb').write(data) 
    else: 
        print("no download necessary")
    predictor = dlib.shape_predictor(modelname)
    aligned_image = align_face(filepath=args.content, predictor=predictor)
    return aligned_image


# entry point to the code
if __name__ == "__main__":
    device = get_default_device()

    parser = TestOptions()
    args = parser.parse() # this method actually prints out all the arguments
    print('*'*98)
    
    # transforming the data before putting it through the model
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])
    
    generator = DualStyleGAN(1024, 512, 8, 2, res_index=6) # TODO: look at the generator code next
    generator.eval() # in evaluation mode, doesn't keep track of gradients or anything

    # loading all the weights - i.e. the pre-trained model 
    # TODO: look at the 2nd argument for the torch.load() method -> what is map_location
    ckpt = torch.load(os.path.join(args.model_path, args.style, args.model_name), map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"]) # moving the trained weights into the generator
    generator = generator.to(device) # moving to GPU

    # not fully sure what is happening here? Loading in another model - the encoder?
    # I think the encoder is the thing that "encodes" the actual user image
    # TODO: investigate this section of code a little bit more
    ###########################################################################################
    model_path = os.path.join(args.model_path, 'encoder.pt')
    
    ckpt = torch.load(model_path, map_location='cpu') # checkpoint
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts) # passing in keyword arguments
    opts.device = device
    encoder = pSp(opts) # (Pixel2style2pixel) pSp is the thing that generates the intermediate image
    encoder.eval()
    encoder.to(device)
    ##############################################################################################

    # some sort of numpy array -> something about an extrinsic style?
    exstyles = np.load(os.path.join(args.model_path, args.style, args.exstyle_name), allow_pickle='TRUE').item()
    # these are all the reference style images in the form of a numpy dictionary
    #print(exstyles['Cartoons_00440_04.jpg'].shape) # each one of these image is of shape (1, 18, 512)

    print('Loaded models successfully!')
    ###########################################################################################

    # disable gradient calculation
    with torch.no_grad():
        viz = []
        # load content image
        if args.align_face: # this is False by default TODO: investigate what this does exactly
            I = transform(run_alignment(args)).unsqueeze(dim=0).to(device)
            I = F.adaptive_avg_pool2d(I, 1024)
        else:
            I = load_image(args.content).to(device) # converts to Tensor, normalises and then moves to the GPU
        viz += [I] # list contains the image

        # reconstructed content image and its intrinsic style code
        # I think the encoder is the thing that does the pre-processing for the image - change eye reflection, modify smile etc.

        # 256 is the output dimension
        img_rec, instyle = encoder(F.adaptive_avg_pool2d(I, 256), randomize_noise=False, return_latents=True, 
                                z_plus_latent=True, return_z_plus_latent=True, resize=False)    
        
        # print(instyle.shape) # the latent code that approximates the input headshot
        # print(img_rec.shape) # the intermediate face -> img_rec = image reconstructed
        # Clamps all elements in input into the range [ min, max ]. Letting min_value and max_value be min and max, respectively,
        img_rec = torch.clamp(img_rec.detach(), -1, 1) # I think img_rec stands for image reconstructed maybe?
        viz += [img_rec]

        ## From this point onwards, the input headshot has been loaded in #######
        ## The cartoon images need to loaded and then passed to the generator ####

        # using the user-provided cartoon images
        if args.cartoon_image_1 is not None:
            latent = load_cartoon_image(os.path.join(args.cartoon_path, args.cartoon_image_1)).to(device)         
            latent = latent.reshape((1,18,512))

            save_name = args.name+'_%s_%s'%(args.cartoon_image_1, os.path.basename(args.content).split('.')[0])

            # TODO: need to resize these images to 1024x1024 otherwise the save_image function will complain
            S1 = load_viz_image(os.path.join(args.cartoon_path, args.cartoon_image_1)).to(device)
            viz += [S1]
            if args.cartoon_image_2 is not None:
                latent1 = load_cartoon_image(os.path.join(args.cartoon_path, args.cartoon_image_2)).to(device)         
                latent1 = latent1.reshape((1,18,512))

                S2 = load_viz_image(os.path.join(args.cartoon_path, args.cartoon_image_2)).to(device)
                viz += [S2]

                # TODO: allow the user to specify later how much of each they want in the output image
                latent = latent.add(latent1) # addition of two tensors
                latent = latent / 2;

                save_name = args.name+'_%s_%s_%s'%(args.cartoon_image_1, args.cartoon_image_2, os.path.basename(args.content).split('.')[0])
                
        else: # this is the default code that uses the existing training images
            stylename = list(exstyles.keys())[args.style_id] # selecting the reference style image
            latent = torch.tensor(exstyles[stylename]).to(device) 
            save_name = args.name+'_%d_%s'%(args.style_id, os.path.basename(args.content).split('.')[0])

            # load style image if it exists
            if os.path.exists(os.path.join(args.data_path, args.style, 'images/train', stylename)):
                S = load_image(os.path.join(args.data_path, args.style, 'images/train', stylename)).to(device)
                viz += [S]

            if args.style_id_2 is not None:
                stylename2 = list(exstyles.keys())[args.style_id_2]
                latent2 = torch.tensor(exstyles[stylename]).to(device)

                latent = latent.add(latent2) # addition of two tensors
                latent = latent / 2;

                if os.path.exists(os.path.join(args.data_path, args.style, 'images/train', stylename)):
                    S2 = load_image(os.path.join(args.data_path, args.style, 'images/train', stylename)).to(device)
                    viz += [S2]

                save_name = args.name+'_%s_%s_%s'%(args.style_id, args.style_id_2, os.path.basename(args.content).split('.')[0])


        if args.preserve_color: #TODO: investigate - what is preserve_color?
            latent[:,7:18] = instyle[:,7:18]
        # extrinsic style code
        # TODO: what exactly is the line below doing in terms of reshaping the vector
        exstyle = generator.generator.style(latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])).reshape(latent.shape)

        # style transfer - WHERE the image is generated
        # input_is_latent: instyle is not in W space
        # z_plus_latent: instyle is in Z+ space
        # use_res: use extrinsic style path, or the style is not transferred
        # interp_weights: weight vector for style combination of two paths

        # [instyle] is the intrinsic style code generated from the Pixel2Style2Pixel encoder
        img_gen, _ = generator([instyle], exstyle, input_is_latent=False, z_plus_latent=True,
                              truncation=args.truncation, truncation_latent=0, use_res=True, interp_weights=args.weight)
        img_gen = torch.clamp(img_gen.detach(), -1, 1)
        viz += [img_gen]

    print('Generated images successfully!')
    
    # saving the generated images to disk
    save_image(torchvision.utils.make_grid(F.adaptive_avg_pool2d(torch.cat(viz, dim=0), 256), 5, 2).cpu(), 
               os.path.join(args.output_path, save_name+'.jpg'))
    # save_image(img_gen[0].cpu(), os.path.join(args.output_path, save_name+'.jpg'))

    print('Saved images successfully!')



#print(stylename)
# print(exstyles.keys())
# print(stylename)

# Instead of exstyles[stylename], provide your own image by using OpenCV or PIL, and then converting into the the same shape as 
# exstyles[stylename]
#(1, 18, 512) shape of tensor

# PIL image
# latent = load_image_grey('./data/cartoon/images/Vanellope.jpeg').to(device);
# print(latent.shape)
# print(x.shape)
# x = x.crop((left, top, right, bottom))

# latent = load_image_grey("/home/sai_k/DualStyleGAN/data/cartoon/images/train/Vanellope.jpeg").to(device) # provide any other type of image
# latent = latent.reshape((1,18,512))
# latent1 = torch.tensor(exstyles["Cartoons_00189_01.jpg"]).to(device)

# latent_avg = np.average([latent, latent1], axis=0)
# latent_avg = torch.mean([latent, latent1], axis=0)

# latent = latent.add(latent1) # addition of two tensors
# avg = torch.mean(pt_addition_result_ex) # mean of output tensors

# latent = latent /2;

# print(latent.size())
# print(exstyles[stylename].shape)
# latent = torch.tensor(x).to(device) # provide any other type of image

#print(latent.shape)
# read as grayscale
# from the top, crop the image
# 18x512 = 9216
# 96x96 = 9216

#(1, 18, 512) shape of tensor

# latent = load_cartoon_image("/home/sai_k/DualStyleGAN/cartoon_faces/moana.jpg").to(device)
# latent = latent.reshape((1,18,512))
# print(latent.size())

# latent = load_cartoon_image("/home/sai_k/DualStyleGAN/cartoon_faces/barney.jpeg").to(device)
# latent = latent.reshape((1,18,512))
# print(latent1.size())

# latent = latent.add(latent1) # addition of two tensors
# latent = latent/2 # taking an average
# print(latent.size())


# load style image if it exists (and add it to the list)
# S = None
# if os.path.exists(os.path.join(args.data_path, args.style, 'images/train', stylename)):

# S1 = load_image_new("/home/sai_k/DualStyleGAN/cartoon_faces/moana.jpg").reshape((1, 18, 512)).to(device)
# print(S1.size())
# S2 = load_image_new("/home/sai_k/DualStyleGAN/cartoon_faces/barney.jpeg").reshape((1, 18, 512)).to(device)
# viz += [S1, S2]

# S = None
# if os.path.exists(os.path.join(args.data_path, args.style, 'images/train', stylename)):
# S = load_image(os.path.join(args.data_path, args.style, 'images/train', "Vanellope.jpg")).to(device)
# S = load_image("/home/sai_k/DualStyleGAN/data/cartoon/images/train/Vanellope.jpeg").reshape((1, 18, 512)).to(device)
# viz += [S]
# vis += [load_image(os.path.join(args.data_path, args.style, "images/Vanellope.jpeg")).to(device)]