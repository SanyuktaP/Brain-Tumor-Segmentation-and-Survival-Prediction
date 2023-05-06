# Brain Tumor Segmentation and Survival Prediction

## Introduction
This project aims to segment brain tumor regions from MRI scans and predict survival using machine learning algorithms.

## Prerequisites
- Python 3.x
- Required packages (can be installed via pip): numpy, pandas, matplotlib, scikit-learn, tensorflow, keras, nibabel, Flask

## Usage
1. Clone the repository
2. Install required packages
3. Run `app.py` to start the Flask web application
4. Open the web application in your browser by navigating to http://localhost:5000/

## Features
- Accepts input of a single multichannel image of MRI and age
- Segments tumor regions using a 3D U-Net model
- Predicts survival time using a linear regression model
- Provides a user interface for easy interaction
- Supports deployment using Docker

## Data
The MRI images used in this project were obtained from the BraTS (Multimodal Brain Tumor Segmentation) challenge dataset.

## Model Training
- Preprocessing: The Training dataset is converted to NPY images for easy segmentation process.
- Segmentation: 3D U-Net model was trained using the preprocessed data.
- Survival Prediction: Linear regression model was trained using the preprocessed data.

## User Interface
The user interface for this project was created using Flask, HTML, and CSS. It allows the user to input a single multichannel MRI NPY image and age, and displays the segmented tumor regions and survival prediction.

## Deployment
This project can be deployed using Docker. A `Dockerfile` is included in the repository.



