# Human Activity Recognition

The goal this project is to perform Human Activity Recognition (HAR) using deep learning techniques, specifically Multilayer Perceptron (MLP) and 1D Convolutional Neural Network (1D CNN). The dataset consists of time series data from smartphone accelerometers, including x, y, and z-axis acceleration data, timestamps, and person IDs. The task is to classify six types of activities: Walking, Jogging, Sitting, Standing, Climbing Upstairs, and Going Downstairs.

## Models

### Multilayer Perceptron (MLP)
- Consists of a flatten layer, four fully connected layers with ReLU activation functions, and a final output layer for class probabilities.

### 1D Convolutional Neural Network (1D CNN)
- Consists of four 1D convolutional layers with ReLU activation functions, a max pooling layer, an adaptive average pooling layer, a dropout layer to prevent overfitting, and a fully connected output layer.

## Evaluation

The models are evaluated based on their accuracy in classifying the activities. A confusion matrix is used to analyze the performance and understand the strengths and weaknesses of each model.

## Experimental Setup

Experiments include hyperparameter tuning, data augmentation techniques, baseline comparison, batch normalization and layer normalization, experimentation with optimizers, model scaling, and sampling strategies to address class imbalance.
