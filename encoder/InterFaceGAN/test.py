import numpy as np

# path = "/dcs/20/u2004267/CS310/DualStyleGAN/facial_editing/boundaries/stylegan_celebahq_pose_boundary.npy"
path = "/home/sai_k/DualStyleGAN/encoder/latent_codes/female_face.npy"
boundary = None

boundary = np.load(path, allow_pickle=True)

if boundary is None:
    print("Boundary has not been loaded in yet")
else:
    print(boundary.shape)
    # v = boundary.copy()
    # v = v/np.linalg.norm(v)
    # print(v[0][1::20])
    # print(np.linalg.norm(v))


# python facial_editing/edit.py --model_name stylegan_celebahq --output_dir facial_editing/new_results/check_if_working_2 --boundary_path facial_editing/boundaries/stylegan_celebahq_smile_boundary.npy --input_latent_codes_path /home/sai_k/DualStyleGAN/facial_editing/instyle_new.npy --steps 4
