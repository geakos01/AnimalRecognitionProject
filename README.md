# Animal Classifier Jupyter Notebook

## Overview

This Jupyter Notebook contains the code for building, training, and evaluating an animal classifier using transfer learning with the DenseNet121 model. The notebook includes data preprocessing, augmentation, and hyperparameter optimization.

## Content

1. **Introduction**
    - Overview of the notebook's purpose and functionality.

2. **Load Config**
    - Definition of the `Config` class and loading necessary configurations.

3. **Check Available Images**
    - Checking the availability of images for each animal category.

4. **Preprocess Images**
    - Preprocessing images and saving processed arrays to the drive.

5. **Load Preprocessed Data and Create X, y for the Model**
    - Loading preprocessed data and creating training and testing datasets.

6. **Data Augmentation**
    - Implementing data augmentation using ImageDataGenerator.

7. **Transfer Learning**
    - Training a transfer learning model using DenseNet121.

8. **Evaluate Model**
    - Evaluating the model on the testing dataset and visualizing results.

9. **Hyperparameter Search Including Architecture**
    - Hyperparameter tuning using Keras Tuner.

10. **Data Overview**
    - Information about the dataset used for training and testing the model.

11. **Additional Notes**
    - Any additional notes or insights about the code.

## Usage

1. Open the notebook in a Jupyter environment (e.g., Google Colab).
2. Execute each cell sequentially.

## Dependencies

- TensorFlow
- Keras
- NumPy
- OpenCV
- Matplotlib
- Seaborn
- Pandas
- PIL (Pillow)
- scikit-learn
- keras_tuner
- hyperopt

## Data Overview

The dataset consists of images belonging to various animal categories. The following animals are included:

- Butterfly
- Cat
- Chicken
- Cow
- Dog
- Elephant
- Horse
- Sheep
- Spider
- Squirrel
- Zebra
- Rhino
- Buffalo

The images are preprocessed and augmented to train the animal classifier model.


**Author:** Akos Gergely