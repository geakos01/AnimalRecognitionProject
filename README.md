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
10. **Computational Resources**

## About the Data

This project utilizes labeled animal images with high usability obtained from two Kaggle datasets:
1. [Animals10 Dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
2. [Animal Classification Dataset](https://www.kaggle.com/datasets/ayushv322/animal-classification?rvi=1)

Images were gathered from 13 classes, with 1000 images collected from each class.

## Selecting Model

Various models, including [VGG16](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16),
[Xception](https://www.tensorflow.org/api_docs/python/tf/keras/applications/xception/Xception),
and [Dense121](https://www.tensorflow.org/api_docs/python/tf/keras/applications/densenet/DenseNet121), were considered. Due to computational limitations, [Dense121](https://www.tensorflow.org/api_docs/python/tf/keras/applications/densenet/DenseNet121) was chosen for further experimentation.

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
- L2 regularization in dense layers
- Learning rate

The best model was selected based on validation accuracy.
Number of dense layers was 3 with the following structure:
- Neurons: [64, 448, 448]
- Regularization: [1.0674541740374033e-05, 0.0007109784706913827, 0.00010575697475283587]

The learning rate was set to 0.00012322526568996265

## Error Investigation

Upon finding the best model, error analysis was conducted by plotting problematic images and examining the confusion matrix.

## Results

The model demonstrated a test set accuracy surpassing 96.3%. Some errors in 
predictions were identified, attributed to misclassifications in the
human-labeled data. Notably, top-performing models on Kaggle achieved 98.03% accuracy when
evaluated on the validation data from the 
[Animals10 Dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
and 88.1% accuracy on the 
[Animal Classification Dataset](https://www.kaggle.com/datasets/ayushv322/animal-classification?rvi=1).

## Create Prototype

The best model was saved and used in the `streamlit_demo.py` program. Users can upload an image of an animal, and the program will predict its class.

## Computational Resources

The model was trained on a Google Colab Pro instance with the following specifications:
- T4 GPU
- 51 GB RAM

The training process took approximately 6 minutes.

## Author

Akos Gergely
