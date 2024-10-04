# -*- coding: utf-8 -*-
"""part1_code

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jhURZ5X8SiXm2PRMik0ZgGacKYtIqiD1

### Part 1: Fitting a 2D Image
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time
import gdown

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

url = "https://drive.google.com/file/d/1rD1aaxN8aSynZ8OPA7EI3G936IF0vcUt/view?usp=sharing"
gdown.download(url=url, output='starry_night.jpg', quiet=False, fuzzy=True)

# Load painting image
painting = imageio.imread("starry_night.jpg")
painting = torch.from_numpy(np.array(painting, dtype=np.float32)/255.).to(device)
height_painting, width_painting = painting.shape[:2]

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
      sin_enc = torch.sin((2**i)*np.pi*x)
      cos_enc = torch.cos((2**i)*np.pi*x)
      results.append(sin_enc)
      results.append(cos_enc)
    #############################  TODO 1(a) END  ##############################
    return torch.cat(results, dim=-1) #dim =-1 here means concatenate along last dimension, example: if tensor contains ([N,D],[N,2D],[N,2D]), it concatenates along last dimension hence it will output tensor of form [N,4D+D]

class model_2d(nn.Module):

    """
    Define a 2D model comprising of three fully connected layers,
    two relu activations and one sigmoid activation.
    """

    def __init__(self, filter_size=128, num_frequencies=6):
        super().__init__()
        #############################  TODO 1(b) BEGIN  ############################
        # for autograder compliance, please follow the given naming for your layers
        self.layer_in = nn.Linear(2*2*num_frequencies+2, filter_size)
        self.layer = nn.Linear(filter_size,filter_size)
        self.layer_out = nn.Linear(filter_size,3)
        self.relu = nn.ReLU
        self.sigmoid = nn.Sigmoid

        #############################  TODO 1(b) END  ##############################

    def forward(self, x):
        #############################  TODO 1(b) BEGIN  ############################
        # example of forward through a layer: y = self.layer_in(x)
        x = F.relu(self.layer_in(x))
        x = F.relu(self.layer(x))
        x = F.sigmoid(self.layer_out(x))


        #############################  TODO 1(b) END  ##############################
        return x

def normalize_coord(height, width, num_frequencies=6):

    """
    Creates the 2D normalized coordinates, and applies positional encoding to them

    Args:
    height (int): Height of the image
    width (int): Width of the image
    num_frequencies (optional, int): The number of frequencies used in
      the positional encoding (default: 6).

    Returns:
    (torch.Tensor): Returns the 2D normalized coordinates after applying positional encoding to them.
    """

    #############################  TODO 1(c) BEGIN  ############################
    # Create the 2D normalized coordinates, and apply positional encoding to them
    x_coords = torch.linspace(0,1,width+1)[:-1]
    y_coords = torch.linspace(0,1,height+1)[:-1]
    x,y = torch.meshgrid(x_coords,y_coords)

    coords = torch.stack((x.flatten(), y.flatten()), dim=-1)

    embedded_coordinates = positional_encoding(coords, num_frequencies)

    #############################  TODO 1(c) END  ############################
    #return x_coords, y_coords,x,y,coords
    return embedded_coordinates

def train_2d_model(test_img, num_frequencies, device, model=model_2d, positional_encoding=positional_encoding, show=True):

    # Optimizer parameters
    lr = 5e-4
    iterations = 10000
    height, width = test_img.shape[:2]

    # Number of iters after which stats are displayed
    display = 2000

    # Define the model and initialize its weights.
    model2d = model(num_frequencies=num_frequencies)
    model2d.to(device)

    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    model2d.apply(weights_init)

    #############################  TODO 1(c) BEGIN  ############################
    # Define the optimizer
    optimizer = torch.optim.Adam(model2d.parameters(),lr)
    #############################  TODO 1(c) END  ############################

    # Seed RNG, for repeatability
    seed = 5670
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Lists to log metrics etc.
    psnrs = []
    iternums = []

    t = time.time()
    t0 = time.time()

    #############################  TODO 1(c) BEGIN  ############################
    # Create the 2D normalized coordinates, and apply positional encoding to them
    x_coords = torch.linspace(0,1,width+1)[:-1]
    y_coords = torch.linspace(0,1,height+1)[:-1]
    x,y = torch.meshgrid(x_coords,y_coords)

    coords = torch.stack((x.flatten(), y.flatten()), dim=-1)

    embedded_coordinates = positional_encoding(coords, num_frequencies)

    #############################  TODO 1(c) END  ############################

    for i in range(iterations+1):
        optimizer.zero_grad()
        #############################  TODO 1(c) BEGIN  ############################
        # Run one iteration
        pred = model2d(embedded_coordinates)
        pred = pred.reshape((height,width,3))

        # Compute mean-squared error between the predicted and target images. Backprop!
        loss1 = nn.MSELoss()
        loss = loss1(pred,test_img.to(device))
        loss.backward()
        optimizer.step()


        #############################  TODO 1(c) END  ############################

        # Display images/plots/stats
        if i % display == 0 and show:
            #############################  TODO 1(c) BEGIN  ############################
            # Calculate psnr
            R = coords[-1][-1]
            psnr = 10*torch.log10_(R**2/loss)
            #############################  TODO 1(c) END  ############################

            print("Iteration %d " % i, "Loss: %.4f " % loss.item(), "PSNR: %.2f" % psnr.item(), \
                "Time: %.2f secs per iter" % ((time.time() - t) / display), "%.2f secs in total" % (time.time() - t0))
            t = time.time()

            psnrs.append(psnr.item())
            iternums.append(i)

            plt.figure(figsize=(13, 4))
            plt.subplot(131)
            plt.imshow(pred.detach().cpu().numpy())
            plt.title(f"Iteration {i}")
            plt.subplot(132)
            plt.imshow(test_img.cpu().numpy())
            plt.title("Target image")
            plt.subplot(133)
            plt.plot(iternums, psnrs)
            plt.title("PSNR")
            plt.show()

    print('Done!')
    torch.save(model2d.state_dict(),'model_2d_' + str(num_frequencies) + 'freq.pt')
    plt.imsave('van_gogh_' + str(num_frequencies) + 'freq.png',pred.detach().cpu().numpy())
    return pred.detach().cpu()