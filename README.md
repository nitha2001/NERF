# Neural_Representations_for_Scene_Fitting_and_Novel_View_Synthesis
This repository contains the implementation for my final project in the course CIS 580: Machine Perception. The project explores Neural Radiance Fields (NeRF) and positional encoding techniques to achieve 2D image fitting and 3D scene reconstruction, as well as synthesizing novel views using neural networks.

## Overview
The project is divided into two main parts:

- **2D Image Fitting**
We first tackle the problem of image representation using a Multilayer Perceptron (MLP) and positional encoding. The goal is to fit a given 2D image by learning a neural representation, which captures the fine details of the image across spatial coordinates.

- **3D Scene Fitting and Novel View Rendering**
In this part, we extend the neural representation from 2D to 3D scenes using the NeRF framework. By sampling rays and applying volume rendering, we aim to fit a 3D scene and generate novel views of the scene.

## Project Components

- **2D Image Fitting with MLPs**
The image fitting pipeline uses a standard MLP to map 2D spatial coordinates to RGB pixel values. The addition of positional encoding enhances the network's ability to capture high-frequency details.

- **NeRF for 3D Scene Reconstruction**
For 3D scene fitting, rays are sampled through the scene to compute densities and colors along the rays. The model then integrates these values using volume rendering to generate a final rendered image.

- **Volume Rendering**
Volume rendering is employed to synthesize the final images by integrating along sampled rays in the scene. This approach allows for realistic image generation from a neural representation of the scene.

## Key Features
- **Positional Encoding**
A critical component for representing high-frequency details in images and scenes. Positional encoding enables the MLP to generalize to fine-grained spatial features.

- **Ray Sampling and Integration**
In the 3D part of the project, rays are cast through the scene, and samples are taken along each ray. These samples are used to estimate color and density, which are then integrated using volume rendering to produce the final image.

- **Rendering Novel Views**
The neural representation allows for rendering novel views of the 3D scene from different perspectives, demonstrating the power of NeRF for photorealistic scene synthesis.

## Associated Files and Functions
**Part 1 - 2D Image Fitting:**
- `positional_encoding()`: Encodes input using sine and cosine functions.
model_2d: Defines a simple 2D neural network model with fully connected layers, ReLU, and sigmoid activations.
- `normalize_coord()`: Generates and normalizes 2D coordinates, applies positional encoding.
train_2d_model(): Trains the 2D model using MSE loss and saves the results. It also plots loss metrics and model predictions.

**Part 2 - 3D Scene Fitting (LEGO Data):**
- `positional_encoding()`: Encodes input using sine and cosine functions to capture different frequency information for higher-dimensional input representation.

- `get_rays()`: Computes the origins and directions of rays for each pixel in the image based on camera intrinsics and the camera pose (extrinsics).

- `stratified_sampling()`: Samples 3D points along the rays between the near and far bounds, adding randomness for better coverage of the scene.

- `volumetric_rendering()`: Performs volumetric rendering by integrating the colors and densities along the rays to form the final 2D image.

- `get_batches()`: Divides the input 3D points and directions into smaller batches and applies positional encoding to ensure efficient processing during model inference.

- `one_forward_pass()`: Executes a single forward pass through the model, calculating rays, sampling 3D points, and rendering the scene to generate an image from the given camera pose.

- `plot_all_poses()`: Visualizes or processes camera poses in 3D space to better understand the position and orientation of the camera relative to the scene.

## Results

<img width="510" alt="Network Architecture" src="https://github.com/user-attachments/assets/ed4b6f22-9f68-422d-afe8-ab177ee50725">

![iter 0](https://github.com/user-attachments/assets/279f8537-4fa1-4c94-b779-f91cafd89bf7)

![it 1000](https://github.com/user-attachments/assets/adf41859-c0e0-4d71-8797-919453262132)

![it 2000](https://github.com/user-attachments/assets/2d3dcefb-79fc-4e60-a1da-afbfc6039330)

![it 3000](https://github.com/user-attachments/assets/6beafbc0-a554-4bf6-aa3b-c05f7732e329)




