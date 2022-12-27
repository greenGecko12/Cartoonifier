import os
import numpy as np
import torch
from util import save_image, load_image
import argparse
from argparse import Namespace
from torchvision import transforms
from torch.nn import functional as F
import torchvision
from model.dualstylegan import DualStyleGAN
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
        self.parser.add_argument("--truncation", type=float, default=0.75, help="truncation for intrinsic style code (content)")
        self.parser.add_argument("--weight", type=float, nargs=18, default=[0.75]*7+[1]*11, help="weight of the extrinsic style")
        self.parser.add_argument("--name", type=str, default='cartoon_transfer', help="filename to save the generated images")
        self.parser.add_argument("--preserve_color", action="store_true", help="preserve the color of the content image")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path of the saved models")
        self.parser.add_argument("--model_name", type=str, default='generator.pt', help="name of the saved dualstylegan")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="path of the output images")
        self.parser.add_argument("--data_path", type=str, default='./data/', help="path of dataset")
        self.parser.add_argument("--align_face", action="store_true", help="apply face alignment to the content image")
        self.parser.add_argument("--exstyle_name", type=str, default=None, help="name of the extrinsic style codes")

        """
        What is the extrinsic style?
        """

    def parse(self):
        # parsing and returning all the arguments passed in by the user
        self.opt = self.parser.parse_args()
        if self.opt.exstyle_name is None:
            if os.path.exists(os.path.join(self.opt.model_path, self.opt.style, 'refined_exstyle_code.npy')):
                self.opt.exstyle_name = 'refined_exstyle_code.npy'
            else:
                self.opt.exstyle_name = 'exstyle_code.npy'     
        args = vars(self.opt) # converting to a dictionary
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
    
"""
WE ARE NOT IN THE CLASS ANYMORE!
"""
def run_alignment(args):
    # pre-processing, just focusing on the face and then cropping everything else
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

# check if we have a GPU and if so, we use that
# otherwise is done on the CPU
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def load_image_grey(filename):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5]),
    ])
    
    img = Image.open(filename).convert('L')
    img = transform(img)
    return img.unsqueeze(dim=0) 

# entry point to the code
if __name__ == "__main__":
    device = "cuda" # change later on to use get_default_device()

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
    ckpt = torch.load(os.path.join(args.model_path, args.style, args.model_name), map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"]) # moving the trained weights into the generator?
    generator = generator.to(device) # moving to GPU

    # not fully sure what is happening here? Loading in another model - the encoder?
    # I think the encoder is the thing that "encodes" the actual user image
    ###########################################################################################
    model_path = os.path.join(args.model_path, 'encoder.pt')
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts) # passing in keyword arguments
    opts.device = device
    encoder = pSp(opts) # TODO: what is pSp?
    encoder.eval()
    encoder.to(device)

    # some sort of numpy array -> something about an extrinsic style?
    exstyles = np.load(os.path.join(args.model_path, args.style, args.exstyle_name), allow_pickle='TRUE').item()
    print(exstyles['Cartoons_00440_04.jpg'].shape)

    print('Load models successfully!')
    ###########################################################################################

    # disable gradient calculation
    with torch.no_grad():
        viz = []
        # load content image
        if args.align_face:
            I = transform(run_alignment(args)).unsqueeze(dim=0).to(device)
            I = F.adaptive_avg_pool2d(I, 1024)
        else:
            I = load_image(args.content).to(device)
        viz += [I] # list contains the image

        # reconstructed content image and its intrinsic style code
        img_rec, instyle = encoder(F.adaptive_avg_pool2d(I, 256), randomize_noise=False, return_latents=True, 
                                z_plus_latent=True, return_z_plus_latent=True, resize=False)    
        img_rec = torch.clamp(img_rec.detach(), -1, 1)
        viz += [img_rec] # adding another thing to the list? some sort of modification to the image maybe?

        stylename = list(exstyles.keys())[args.style_id]

        # Instead of exstyles[stylename], provide your own image by using OpenCV or PIL, and then converting into the the same shape as 
        # exstyles[stylename]
        #(1, 18, 512) shape of tensor
       
        # PIL image
        x = load_image_grey('test2.png')
        #print(x.shape)
        x = x.reshape((1,18,512))
        # print(x.shape)
        # x = x.crop((left, top, right, bottom))

        # latent = torch.tensor(exstyles[stylename]).to(device) # provide any other type of image

        # print(exstyles[stylename].shape)
        latent = torch.tensor(x).to(device) # provide any other type of image
        #print(latent.shape)
        # read as grayscale
        # from the top, crop the image
        # 18x512 = 9216
        # 96x96 = 9216

        if args.preserve_color:
            latent[:,7:18] = instyle[:,7:18]
        # extrinsic style code
        exstyle = generator.generator.style(latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])).reshape(latent.shape)

        # load style image if it exists
        S = None
        if os.path.exists(os.path.join(args.data_path, args.style, 'images/train', stylename)):
            S = load_image(os.path.join(args.data_path, args.style, 'images/train', stylename)).to(device)
            viz += [S]

        # style transfer - WHERE the image is generated
        # input_is_latent: instyle is not in W space
        # z_plus_latent: instyle is in Z+ space
        # use_res: use extrinsic style path, or the style is not transferred
        # interp_weights: weight vector for style combination of two paths
        img_gen, _ = generator([instyle], exstyle, input_is_latent=False, z_plus_latent=True,
                              truncation=args.truncation, truncation_latent=0, use_res=True, interp_weights=args.weight)
        img_gen = torch.clamp(img_gen.detach(), -1, 1)
        viz += [img_gen]

    print('Generate images successfully!')
    
    # saving the generated images to disk
    save_name = args.name+'_%d_%s'%(args.style_id, os.path.basename(args.content).split('.')[0])
    save_image(torchvision.utils.make_grid(F.adaptive_avg_pool2d(torch.cat(viz, dim=0), 256), 4, 2).cpu(), 
               os.path.join(args.output_path, save_name+'_overview.jpg'))
    save_image(img_gen[0].cpu(), os.path.join(args.output_path, save_name+'.jpg'))

    print('Save images successfully!')
