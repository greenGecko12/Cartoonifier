import numpy as np

path = "/dcs/20/u2004267/CS310/DualStyleGAN/facial_editing/boundaries/stylegan_celebahq_pose_boundary.npy"
boundary = None

boundary = np.load(path)

if boundary is None:
    print("Boundary has not been loaded in yet")
else:
    print(boundary)