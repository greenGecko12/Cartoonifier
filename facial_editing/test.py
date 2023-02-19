import numpy as np

# path = "/dcs/20/u2004267/CS310/DualStyleGAN/facial_editing/boundaries/stylegan_celebahq_pose_boundary.npy"
path = "/dcs/20/u2004267/CS310/DualStyleGAN/facial_editing/boundaries/stylegan_celebahq_gender_w_boundary.npy"
boundary = None



boundary = np.load(path)

if boundary is None:
    print("Boundary has not been loaded in yet")
else:
    print(boundary)
    # v = boundary.copy()
    # v = v/np.linalg.norm(v)
    # print(v[0][1::20])
    # print(np.linalg.norm(v))