# Neural_Representations_for_Scene_Fitting_and_Novel_View_Synthesis
This repository contains the implementation for my final project in the course CIS 580: Machine Perception. The project explores Neural Radiance Fields (NeRF) and positional encoding techniques to achieve 2D image fitting and 3D scene reconstruction, as well as synthesizing novel views using neural networks.

## Overview
The project is divided into two main parts:

**2D Image Fitting**
We first tackle the problem of image representation using a Multilayer Perceptron (MLP) and positional encoding. The goal is to fit a given 2D image by learning a neural representation, which captures the fine details of the image across spatial coordinates.

**3D Scene Fitting and Novel View Rendering**
In this part, we extend the neural representation from 2D to 3D scenes using the NeRF framework. By sampling rays and applying volume rendering, we aim to fit a 3D scene and generate novel views of the scene.

## Project Components

**2D Image Fitting with MLPs**
The image fitting pipeline uses a standard MLP to map 2D spatial coordinates to RGB pixel values. The addition of positional encoding enhances the network's ability to capture high-frequency details.

**NeRF for 3D Scene Reconstruction**
For 3D scene fitting, rays are sampled through the scene to compute densities and colors along the rays. The model then integrates these values using volume rendering to generate a final rendered image.

**Volume Rendering**
Volume rendering is employed to synthesize the final images by integrating along sampled rays in the scene. This approach allows for realistic image generation from a neural representation of the scene.

## Key Features
**Positional Encoding**
A critical component for representing high-frequency details in images and scenes. Positional encoding enables the MLP to generalize to fine-grained spatial features.

**Ray Sampling and Integration**
In the 3D part of the project, rays are cast through the scene, and samples are taken along each ray. These samples are used to estimate color and density, which are then integrated using volume rendering to produce the final image.

**Rendering Novel Views**
The neural representation allows for rendering novel views of the 3D scene from different perspectives, demonstrating the power of NeRF for photorealistic scene synthesis.
