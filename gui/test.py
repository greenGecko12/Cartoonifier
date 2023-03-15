import numpy as np
from torch import load

# path = "/dcs/20/u2004267/CS310/DualStyleGAN/facial_editing/boundaries/stylegan_celebahq_pose_boundary.npy"
path = "/home/sai_k/DualStyleGAN/gui/modified_latent.npy"
boundary = None

# boundary = np.load(path, allow_pickle=True)
boundary = load("/home/sai_k/DualStyleGAN/smile.pt").numpy()

if boundary is None:
    print("Boundary has not been loaded in yet")
else:
    print(boundary.shape)
    # v = boundary.copy()
    # v = v/np.linalg.norm(v)
    # print(v[0][1::20])
    # print(np.linalg.norm(v))