import os
import gdown
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

url = "https://drive.google.com/file/d/13eBK_LWxs4SkruFKH7glKK9jwQU1BkXK/view?usp=sharing"
gdown.download(url=url, output='lego_data.npz', quiet=False, fuzzy=True)

def positional_encoding(x, num_frequencies=6, incl_input=True):

    """
    Apply positional encoding to the input.

    Args:
    x (torch.Tensor): Input tensor to be positionally encoded.
      The dimension of x is [N, D], where N is the number of input coordinates,
      and D is the dimension of the input coordinate.
    num_frequencies (optional, int): The number of frequencies used in
     the positional encoding (default: 6).
    incl_input (optional, bool): If True, concatenate the input with the
        computed positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """

    results = []
    if incl_input:
        results.append(x)
    #############################  TODO 1(a) BEGIN  ############################
    # encode input tensor and append the encoded tensor to the list of results.
    L = num_frequencies
    for i in range(L):
      sin_enc = torch.sin((2**i)*torch.pi*x)
      cos_enc = torch.cos((2**i)*torch.pi*x)
      results.append(sin_enc)
      results.append(cos_enc)
    #############################  TODO 1(a) END  ##############################
    return torch.cat(results, dim=-1) #dim =-1 here means concatenate along last dimension, example: if tensor contains ([N,D],[N,2D],[N,2D]), it concatenates along last dimension hence it will output tensor of form [N,4D+D]


def get_rays(height, width, intrinsics, w_R_c, w_T_c):

    """
    Compute the origin and direction of rays passing through all pixels of an image (one ray per pixel).

    Args:
    height: the height of an image.
    width: the width of an image.
    intrinsics: camera intrinsics matrix of shape (3, 3).
    w_R_c: Rotation matrix of shape (3,3) from camera to world coordinates.
    w_T_c: Translation vector of shape (3,1) that transforms

    Returns:
    ray_origins (torch.Tensor): A tensor of shape (height, width, 3) denoting the centers of
      each ray. Note that desipte that all ray share the same origin, here we ask you to return
      the ray origin for each ray as (height, width, 3).
    ray_directions (torch.Tensor): A tensor of shape (height, width, 3) denoting the
      direction of each ray.
    """

    device = intrinsics.device
    ray_directions = torch.zeros((height, width, 3), device=device)  # placeholder
    ray_origins = torch.zeros((height, width, 3), device=device)  # placeholder

    #############################  TODO 2.1 BEGIN  ##########################
    coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='xy')
    pix_coords = torch.stack(coords, -1).reshape((-1,2))
    pix_coords = torch.concat((pix_coords, torch.ones(pix_coords.shape[0],1)), -1).to(device)

    pix_to_rays = torch.linalg.inv(intrinsics) @ pix_coords.T
    pix_to_rays = w_R_c @ pix_to_rays

    rays_directions = pix_to_rays.T.reshape((height,width,3))
    rays_origins = torch.broadcast_to(w_T_c.reshape((1,3)), ((height, width,3)))

    #############################  TODO 2.1 END  ############################
    return rays_origins, rays_directions



def stratified_sampling(ray_origins, ray_directions, near, far, samples):

    """
    Sample 3D points on the given rays. The near and far variables indicate the bounds of sampling range.

    Args:
    ray_origins: Origin of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    ray_directions: Direction of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    near: The 'near' extent of the bounding volume.
    far:  The 'far' extent of the bounding volume.
    samples: Number of samples to be drawn along each ray.

    Returns:
    ray_points: Query 3D points along each ray. Shape: (height, width, samples, 3).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).
    """

    #############################  TODO 2.2 BEGIN  ############################
    N = samples
    H,W,_ = ray_origins.shape
    ray_points = torch.zeros((H,W,N,3))
    depth_points = torch.zeros((H,W,N))
    for i in range(N):
      depth = near + (i)*(far-near)/N
      ray = ray_origins + depth*ray_directions
      depth_points[:,:,i] = depth
      ray_points[:,:,i,:] = ray


    #############################  TODO 2.2 END  ############################
    return ray_points, depth_points


class nerf_model(nn.Module):

    """
    Define a NeRF model comprising eight fully connected layers and following the
    architecture described in the NeRF paper.
    """

    def __init__(self, filter_size=256, num_x_frequencies=6, num_d_frequencies=3):
        super().__init__()

        #############################  TODO 2.3 BEGIN  ############################
        # for autograder compliance, please follow the given naming for your layers

        self.layers = nn.ModuleDict({
            'layer_1': nn.Linear(6*num_x_frequencies+3,filter_size),
            'layer_2': nn.Linear(filter_size,filter_size),
            'layer_3': nn.Linear(filter_size,filter_size),
            'layer_4': nn.Linear(filter_size,filter_size),
            'layer_5': nn.Linear(filter_size,filter_size),
            'layer_6': nn.Linear(filter_size+6*num_x_frequencies+3,filter_size),
            'layer_7': nn.Linear(filter_size,filter_size),
            'layer_8': nn.Linear(filter_size,filter_size),
            'layer_s': nn.Linear(filter_size,1),
            'layer_9': nn.Linear(filter_size,filter_size),
            'layer_10': nn.Linear(filter_size+6*num_d_frequencies+3,int(filter_size/2)),
            'layer_11': nn.Linear(int(filter_size/2),3)
        })

        #############################  TODO 2.3 END  ############################


    def forward(self, x, d):
        #############################  TODO 2.3 BEGIN  ############################
        # example of forward through a layer: y = self.layers['layer_1'](x)
        y = F.relu(self.layers['layer_1'](x))
        y = F.relu(self.layers['layer_2'](y))
        y = F.relu(self.layers['layer_3'](y))
        y = F.relu(self.layers['layer_4'](y))
        y = F.relu(self.layers['layer_5'](y))
        y = torch.cat((y,x), dim=-1)
        y = F.relu(self.layers['layer_6'](y))
        y = F.relu(self.layers['layer_7'](y))
        y = F.relu(self.layers['layer_8'](y))
        sigma = self.layers['layer_s'](y)
        y = self.layers['layer_9'](y)
        y = torch.cat((y,d), dim=-1)
        y = F.relu(self.layers['layer_10'](y))
        rgb = F.sigmoid(self.layers['layer_11'](y))
        #############################  TODO 2.3 END  ############################
        return rgb, sigma

def get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies):

    def get_chunks(inputs, chunksize = 2**15):
        """
        This fuction gets an array/list as input and returns a list of chunks of the initial array/list
        """
        return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

    """
    This function returns chunks of the ray points and directions to avoid memory errors with the
    neural network. It also applies positional encoding to the input points and directions before
    dividing them into chunks, as well as normalizing and populating the directions.
    """
    #############################  TODO 2.3 BEGIN  ############################
    #ray_directions_normed = ray_directions/np.linalg.norm(ray_directions)
    # normalize dirns
    ray_directions /= torch.norm(ray_directions, dim=2).unsqueeze(dim=2)

    # populate along each ray
    ray_directions = torch.broadcast_to(ray_directions.unsqueeze(2), ray_points.shape)

    # flatten the vector
    flattened_points = ray_points.reshape(-1,3)
    flattened_dir = ray_directions.reshape(-1,3)

    # positional encoding
    points = positional_encoding(flattened_points,num_x_frequencies)
    dirs = positional_encoding(flattened_dir,num_d_frequencies)

    # helper function and get batches
    ray_points_batches = get_chunks(points)
    ray_directions_batches = get_chunks(dirs)
    #############################  TODO 2.3 END  ############################

    return ray_points_batches, ray_directions_batches


def volumetric_rendering(rgb, s, depth_points):

    """
    Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.

    Args:
    rgb: RGB color at each query location (X, Y, Z). Shape: (height, width, samples, 3).
    sigma: Volume density at each query location (X, Y, Z). Shape: (height, width, samples).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).

    Returns:
    rec_image: The reconstructed image after applying the volumetric rendering to every pixel.
    Shape: (height, width, 3)
    """

    #############################  TODO 2.4 BEGIN  ############################
    deltas = depth_points[...,1:]-depth_points[...,:-1]
    #deltas = torch.cat([deltas, torch.full_like(deltas[..., :1], 1e9)], dim=-1)
    #deltas = torch.concat([deltas, torch.broadcast_to([1e9], shape=(N, H, W, 1))], dim=-1)
    deltas = deltas.to(device)
    deltas = torch.concat((deltas,1e9*torch.ones(deltas.shape[0:2], device=device).unsqueeze(-1)),-1)


    s = F.relu(s)

    exp_t =  torch.exp(-s*deltas)

    transmittance = torch.cumprod(exp_t,dim = -1)
    transmittance = torch.concat((torch.ones(transmittance.shape[0:2], device=device).unsqueeze(-1),transmittance[...,:-1]),-1)

    rec_image = torch.sum(transmittance.unsqueeze(-1)*(1-exp_t.unsqueeze(-1))*rgb, dim=-2 )

    #############################  TODO 2.4 END  ############################

    return rec_image


def one_forward_pass(height, width, intrinsics, pose, near, far, samples, model, num_x_frequencies, num_d_frequencies):

    #############################  TODO 2.5 BEGIN  ############################
    w_R_c = pose[:3,:3]
    w_T_c = pose[:3,3]
    #compute all the rays from the image
    ray_origins,ray_directions = get_rays(height,width,intrinsics,w_R_c,w_T_c)

    #sample the points from the rays
    ray_points,depth_points = stratified_sampling(ray_origins,ray_directions,near,far,samples)

    #divide data into batches to avoid memory errors
    ray_points_batches,ray_directions_batches = get_batches(ray_points,ray_directions,num_x_frequencies,num_d_frequencies)


    #forward pass the batches and concatenate the outputs at the end
    rgb = []
    sigma = []
    for points, dirs in zip(ray_points_batches,ray_directions_batches):
      rgb_batch,sigma_batch = model(points.to(device),dirs.to(device))
      rgb.append(rgb_batch)
      sigma.append(sigma_batch)

    rgb = torch.cat(rgb).reshape(height, width, samples, 3)
    sigma = torch.cat(sigma).reshape(height, width, samples)






    # Apply volumetric rendering to obtain the reconstructed image
    rec_image = volumetric_rendering(rgb,sigma,depth_points)

    #############################  TODO 2.5 END  ############################

    return rec_image

def plot_all_poses(poses):

    #############################  TODO 2.1 BEGIN  ############################
    poses = torch.Tensor(poses).to(device)

    origins = torch.zeros((poses.shape[0],height,width,3))
    directions = torch.zeros_like(origins)

    for i in range(poses.shape[0]):
      w_R_c = torch.tensor(poses[i,:3,:3])
      w_T_c = torch.tensor(poses[i,:3,3])
      origin,direction = get_rays(height,width,intrinsics,w_R_c, w_T_c)
      origins[i] = origin.to('cpu')
      directions[i] = direction.to('cpu')
      

    #############################  TODO 2.1 END  ############################

    ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
    _ = ax.quiver(origins[..., 0].flatten(),
                  origins[..., 1].flatten(),
                  origins[..., 2].flatten(),
                  directions[..., 0].flatten(),
                  directions[..., 1].flatten(),
                  directions[..., 2].flatten(), length=0.12, normalize=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('z')
    plt.show()
