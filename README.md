# Animal Classifier Project

## Overview

The aim of this project is to create a model using transfer learning that can predict animal classes with high accuracy.

## Content

1. **About the Data**
2. **Selecting Model**
3. **Preprocess Images**
4. **Data Augmentation**
5. **Splitting Data**
6. **Transfer Learning**
7. **Hyperparameter Optimization**
8. **Error Investigation**
9. **Creating Prototype**

## About the Data

The data for this project was collected from two Kaggle datasets:
1. [Animals10 Dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
2. [Animal Classification Dataset](https://www.kaggle.com/datasets/ayushv322/animal-classification?rvi=1)

Images were gathered from 13 classes, with 1000 images collected from each class.

## Selecting Model

Various models, including VGG16, Xception, and Dense 121, were considered. Due to computational limitations, Dense 121 was chosen for further experimentation.

## Preprocess Images

To facilitate transfer learning, the images were reformatted. They were reshaped to (224,224), and the Dense 121 preprocessing function was applied to standardize color and perform other preprocessing steps.

## Data Augmentation

Data augmentation was employed to enhance model accuracy. The following operations were applied to create additional images:
- Rotation
- Width shift
- Height shift
- Shear
- Zoom
- Horizontal flip
- Vertical flip

## Splitting Data

The distribution for each class during data splitting was as follows:
- Train: 675
- Validation: 225
- Test: 100

The training set was augmented with additional images.

## Transfer Learning

After creating the train-test split, preprocessing data, and importing the Dense 121 model, transfer learning was performed. The last layer was replaced with additional Dense layers, and the original model's weights were set to untrainable. The output layer has softmax activation, and the number of neurons matches the number of classes.

## Hyperparameter Optimization

After creating a base model, a hyperparameter search was conducted to find the best values for:
- Number of dense layers
- Number of neurons in dense layers
- Regularization in dense layers
- Learning rate

## Error Investigation

Upon finding the best model, error analysis was conducted by plotting problematic images and examining the confusion matrix.

## Results

The model achieved over 96.3% accuracy on the test set. Some errors were attributed to human label misclassifications.

## Create Prototype

The best model was saved and used in the `streamlit_demo.py` program. Users can upload an image of an animal, and the program will predict its class.

## Author

Akos Gergely
