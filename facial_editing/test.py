import numpy as np

# path = "/dcs/20/u2004267/CS310/DualStyleGAN/facial_editing/boundaries/stylegan_celebahq_pose_boundary.npy"
# path = "/dcs/20/u2004267/CS310/DualStyleGAN/facial_editing/boundaries/stylegan_celebahq_gender_w_boundary.npy"
# path = "/dcs/20/u2004267/CS310/DualStyleGAN/facial_editing/data/test_face/wp.npy"
# path = "/dcs/20/u2004267/CS310/DualStyleGAN/checkpoint/cartoon/refined_exstyle_code.npy"
# path = "/dcs/20/u2004267/CS310/DualStyleGAN/facial_editing/data/new_face_5/wp.npy"
path = "/dcs/20/u2004267/CS310/DualStyleGAN/facial_editing/data/female_face.npy"
# path_1 = "/dcs/20/u2004267/CS310/DualStyleGAN/facial_editing/instyle_2.npy"
boundary = None

boundary = np.load(path, allow_pickle=True)
# boundary_1 = np.load(path_1, allow_pickle=True)

if boundary is None:
    print("Boundary has not been loaded in yet")
else:
    # print(boundary - boundary_1)
    print(boundary.shape)
    # v = boundary.copy()
    # v = v/np.linalg.norm(v)
    # print(v[0][1::20])
    # print(np.linalg.norm(v))