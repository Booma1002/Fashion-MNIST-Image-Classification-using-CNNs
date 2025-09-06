# Fashion MNIST Image Classification

## Overview
##### This project develops a Convolutional Neural Network (CNN) to classify images of clothing and accessories from the Fashion MNIST dataset. The objective is to build a robust image classifier, employing data augmentation and regularization techniques to achieve high accuracy on a multi-class problem.

## Data
#### The model is trained on the built-in fashion_mnist dataset available through the TensorFlow/Keras library.
-> https://github.com/zalandoresearch/fashion-mnist

## Process
#### Data Exploration: 
 Loaded the dataset, analyzed image dimensions (28x28 grayscale), and visualized sample images with their corresponding labels.
#### Preprocessing & Augmentation: 
 Reshaped images to include a channel dimension for the CNN. Implemented data augmentation (rotation, shifts, flips) using ImageDataGenerator to create a more robust training set and prevent overfitting.
#### Modeling: 
 Constructed a Convolutional Neural Network (CNN) architecture with multiple convolutional layers, max pooling, and dropout for regularization.
#### Training & Evaluation:
 Trained the model using EarlyStopping to automatically find the optimal number of epochs. Evaluated the final model on the unseen test set.
#### Interpretation:
 Analyzed performance using accuracy/loss plots and a confusion matrix. 
 
## Results
* The final model achieved an accuracy of approximately 84% on the unseen test set.

* Analysis from the confusion matrix and Grad-CAM showed the model performed well but had the most difficulty distinguishing between visually similar categories like 'Shirt', 'T-shirt/top', and 'Pullover'.

## Technologies Used
* Pandas

* NumPy

* TensorFlow & Keras

* Scikit-learn

* Seaborn

* Matplotlib
