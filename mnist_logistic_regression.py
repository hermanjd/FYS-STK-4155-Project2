import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from backpropegation import *
import gzip
import matplotlib.pyplot as plt

num_images = 60000
image_size = 28
neurons = 256
batch_size = 32
epochs = 3
decay=5e-7
learning_rate = 0.02
momentum=0.9

#import of the images
images_file_stream = gzip.open('data/train-images-idx3-ubyte.gz','r')
labels_file_stream = gzip.open('data/train-labels-idx1-ubyte.gz','r')
labels_file_stream.read(8)
buf = labels_file_stream.read(num_images)
labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
b = np.zeros((labels.size, labels.max() + 1))
b[np.arange(labels.size), labels] = 1

images_file_stream.read(16)
buf = images_file_stream.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float64)
data = data.reshape(num_images, image_size*image_size)

#Scaling data to be values between 0 and 1
data = data / data.max()

X_train, X_test, y_train, y_test = train_test_split(data, b, test_size=0.2, random_state=42)

dense1 = Layer_Dense(X_train.shape[1], 10, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(decay=decay, learning_rate=learning_rate, momentum=momentum)

for epoch in range(epochs):
    print(epoch)
    for batch in range(int(len(X_train)/batch_size)):
        x_batch = X_train[batch*batch_size:(batch+1)*batch_size]
        y_batch = y_train[batch*batch_size:(batch+1)*batch_size]
        dense1.forward(x_batch)
        loss = loss_activation.forward(dense1.output, y_batch)
        predictions = np.argmax(loss_activation.output, axis=1)
        y_train_argmax = np.argmax(y_batch, axis=1)
        accuracy = np.mean(predictions==y_train_argmax)
        # Backward pass
        loss_activation.backward(loss_activation.output, y_batch)
        dense1.backward(loss_activation.dinputs)
        # Update weights and biases
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.post_update_params()


dense1.forward(X_test)
loss = loss_activation.forward(dense1.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)
y_test_argmax = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions==y_test_argmax)
print(accuracy)

# Managed to get an accuracy of 0.9162 after 500 generations