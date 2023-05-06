# Brain Tumor Segmentation and Survival Prediction

## Introduction
This project aims to segment brain tumor regions from MRI scans and predict survival using machine learning algorithms.

## Dataset link
https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation/code

## Prerequisites
- Python 3.x
- Required packages (can be installed via pip): numpy, pandas, matplotlib, scikit-learn, tensorflow, keras, nibabel, Flask

## Usage
1. Clone the repository
2. Create a conda environment after opening the repository

```bash
conda create -n brats python=3.8 -y
```

```bash
conda activate brats
```

3. Install required packages
```bash
pip install -r requirements.txt
```

4. Run `app.py` to start the Flask web application
```bash
python app.py
```

5. Open the web application in your browser by navigating to http://localhost:5000/

## Architecture
![full_architecture](https://user-images.githubusercontent.com/97040413/236619684-1278a17c-0b18-4707-877e-0bb834612730.png)

## Features
- Accepts input of a single multichannel image of MRI and age
- Segments tumor regions using a 3D U-Net model
- Predicts survival time using a linear regression model
- Provides a user interface for easy interaction
- Supports deployment using Docker

## Data
The MRI images used in this project were obtained from the BraTS (Multimodal Brain Tumor Segmentation) challenge dataset.
This containes 4 label volumes:
1. T1
2. T2
3. T1ce
4. Flair

These are combined into single multichannel image in data for easy processing.

## Model Training
- Preprocessing: The Training dataset is converted to NPY images for easy segmentation process.
- Segmentation: 3D U-Net model was trained using the preprocessed data.
- Survival Prediction: Linear regression model was trained using the preprocessed data.

## User Interface
The user interface for this project was created using Flask, HTML, and CSS. It allows the user to input a single multichannel MRI NPY image and age, and displays the segmented tumor regions and survival prediction.

## Deployment
This project can be deployed using Docker. A `Dockerfile` is included in the repository.



