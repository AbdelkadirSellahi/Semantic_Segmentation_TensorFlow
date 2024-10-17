# Flood Scene Segmentation Using Unet and ResNet34 (TensorFlow)

This project implements semantic segmentation of flood scenes using the **Unet** architecture with a **ResNet34** backbone, leveraging **TensorFlow** as the deep learning framework. The model is designed to classify and segment UAV-captured flood imagery into various categories, such as flooded buildings, roads, and other objects in flood-affected areas. The goal is to provide a fast and efficient segmentation model that can be trained with limited GPU resources.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
  - [Visualizing Results](#visualizing-results)
- [Customization](#customization)
- [Contributing](#contributing)

## Overview

Flood scene segmentation is important for disaster management and mitigation. Using UAVs (drones) to capture real-time flood imagery, this project offers an automated solution for segmenting various components of flood scenes. The segmentation model classifies each pixel in an image into one of 10 classes, including flooded and non-flooded buildings, roads, water, and vegetation.

### Classes to Identify:
- **Background**
- **Building-flooded**
- **Building-non-flooded**
- **Road-flooded**
- **Road-non-flooded**
- **Water**
- **Tree**
- **Vehicle**
- **Pool**
- **Grass**

## Dataset

The [**FloodNet dataset**](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021?fbclid=IwAR2XIwe5nJg5VSgxgCldM7K0HPtVsDxB0fjd8cJJZfz6WMe3g0Pxg2W3PlE) consists of UAV-captured flood scenes, each accompanied by a corresponding segmentation mask. Each pixel in the mask is labeled as one of 10 classes.

- Images are resized to **128x128** pixels for faster training.
- The dataset is split into **train**, **validation**, and **test** sets.

You can replace this dataset with any other image segmentation dataset, ensuring that the classes and masks are properly formatted.

## Model Architecture

This project uses **Unet** with a **ResNet34** backbone, built on the **TensorFlow** framework:
- **Backbone**: ResNet34 (pre-trained on ImageNet)
- **Input Size**: 128x128 RGB images
- **Output**: 128x128 segmentation mask with 10 classes
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam

The image size is set to **128x128** to enable faster training on lower-end GPUs. However, you can modify these settings for higher-resolution images or more computational power.

## Installation

To run this project in a Jupyter Notebook or Google Colab, you need to install the required dependencies.

1. Clone the repository:

   ```bash
   git clone https://github.com/AbdelkadirSellahi/Semantic_Segmentation_TensorFlow.git
   cd Semantic_Segmentation_TensorFlow
   ```

2. Install the required Python packages:

   ```bash
   pip install tensorflow==2.8.0
   pip install opencv-python-headless
   pip install segmentation-models
   ```

In Colab, you can install the required packages directly using:

```bash
!pip install tensorflow==2.8.0
!pip install opencv-python-headless
!pip install segmentation-models
```

## Usage

Everything is implemented in a single Jupyter Notebook (`Semantic_Segmentation_TensorFlow.ipynb`). You can run this notebook in Colab or any environment that supports GPU.

### Training the Model

To start training, follow these steps:
1. Open the `Semantic_Segmentation_TensorFlow.ipynb` notebook.
2. Follow the steps to load the dataset and run the training.

Default training parameters:
- **Batch Size**: 8
- **Image Size**: 128x128
- **Epochs**: 14

### Evaluating the Model

After training, you can evaluate the model on the test set by running the evaluation cell in the notebook. Key metrics include:
- **Test Accuracy**
- **Mean IoU**: Intersection over Union for each class is calculated using a custom IoU metric.

### Visualizing Results

You can visualize the segmentation results by running the visualization section in the notebook. This will display:
- The original test images.
- The ground truth segmentation masks.
- The predicted segmentation masks.

This helps visually assess how well the model has segmented the images.

## Customization

The notebook allows for easy customization:
- **Image Size**: Increase the image size for better resolution if your GPU can handle it.
- **Batch Size**: Adjust the batch size to suit the available GPU memory.
- **Epochs**: You can train the model for more epochs to improve accuracy.
- **Learning Rate**: Adjust the learning rate for fine-tuning the model's training.

## Contributing

We welcome contributions! If you'd like to add features or fix issues:
1. Fork the repository.
2. Create a new branch.
3. Submit a pull request.

Please provide clear documentation for any changes or new features.

## Authors

- [**ABDELKADIR Sellahi**](https://github.com/AbdelkadirSellahi)
