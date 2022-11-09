import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from backpropegation import *
import gzip
import matplotlib.pyplot as plt

#import of the images
images_file_stream = gzip.open('data/train-images-idx3-ubyte.gz','r')
labels_file_stream = gzip.open('data/train-labels-idx1-ubyte.gz','r')
labels_file_stream.read(8)
buf = labels_file_stream.read(50000)
labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
b = np.zeros((labels.size, labels.max() + 1))
b[np.arange(labels.size), labels] = 1

image_size = 28
num_images = 50000
images_file_stream.read(16)
buf = images_file_stream.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float64)
data = data.reshape(num_images, image_size*image_size)
data = data / data.max()

X_train, X_test, y_train, y_test = train_test_split(data, b, test_size=0.2, random_state=42)

accuracy_ReLU = []
accuracy_Sigmoid = []
accuracy_ReLU_two_layers = []
accuracy_Sigmoid_two_layers = []

neurons = 256

dense1 = Layer_Dense(X_train.shape[1], neurons, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(neurons, neurons,  weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(neurons, 10)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_Adam(decay=5e-7, learning_rate=0.008)
for epoch in range(500):
    dense1.forward(X_train)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    data_loss = loss_activation.forward(dense3.output, y_train)

    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)

    predictions = np.argmax(loss_activation.output, axis=1)
    y_test_argmax = np.argmax(y_train, axis=1)
    accuracy = np.mean(predictions==y_test_argmax)
    # Calculate overall loss
    loss = data_loss + regularization_loss
    print(f'accuracy_ReLU_two_layers: loss:{loss} accuracy:{accuracy}')
    # Backward pass
    loss_activation.backward(loss_activation.output, y_train)
    dense3.backward(loss_activation.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
loss = loss_activation.forward(dense3.output, y_test)
predictions = np.argmax(loss_activation.output, axis=1)
y_test_argmax = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions==y_test_argmax)
print(f'accuracy_ReLU_two_layers: neurons:{neurons} accuracy:{accuracy}')
accuracy_ReLU_two_layers.append(accuracy)

#Might be good to check training and testing accuracy to see if the model is overfit

# Test results 80 Epoch, Optimizer_Adam(decay=5e-7, learning_rate=0.01)
# Training data: 0.97625 - Test Data: 0.9608

# Test results 100 Epoch, Optimizer_Adam(decay=5e-7, learning_rate=0.01)
# Training data: 0.9895 - Test Data: 0.9681

# Test results 500 Epoch, Optimizer_Adam(decay=5e-7, learning_rate=0.008)
# Training data: 0.9974 - Test Data:0.9715

# Note, while the test data accuracy seems to be going up with more epoch, we can very clearly see that the model is also overfitting to the training data due to the 
# high fit of the training set at 99.74% and the testing accuracy at 97.15%. This gives ut a 2.59% differance between training and test accuracy.
