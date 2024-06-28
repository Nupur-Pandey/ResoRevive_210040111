# ResoRevive_210040111

1-Import Libraries

The necessary libraries and modules are imported, including PyTorch, torchvision for datasets and transformations, and matplotlib for plotting.

2-Dataset Preparation

The MNIST dataset is downloaded and loaded using torchvision.datasets. The dataset is split into training and testing sets, and DataLoaders are created for batching and shuffling the data.

3-Model Definition

The Softmax class defines the neural network architecture:

Three fully connected layers (Linear1, Linear2, Linear3).
Batch normalization layers (bn1, bn2).
Dropout layers (droupout1, droupout2).
The output layer is initialized with zero weights and biases.
Forward Pass

The forward method implements the forward pass through the network, applying linear transformations, batch normalization, dropout, and softmax activation.

4-Accuracy Computation

The compute_accuracy function calculates the accuracy of the model on the provided data loader.

5-Training the Model

The model is trained for 25 epochs using the Adam optimizer. The cross-entropy loss is computed, and the weights are updated accordingly. Training accuracy is printed after each epoch.

6-Evaluation

After training, the model parameters and test accuracy are printed
