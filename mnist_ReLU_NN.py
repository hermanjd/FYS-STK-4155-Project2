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
buf = labels_file_stream.read(5000)
labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
b = np.zeros((labels.size, labels.max() + 1))
b[np.arange(labels.size), labels] = 1

image_size = 28
num_images = 5000
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


dense1 = Layer_Dense(X_train.shape[1], 16)
activation1 = Activation_ReLU()
dense3 = Layer_Dense(16, 10)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD()

for epoch in range(10000):
    dense1.forward(X_train)
    activation1.forward(dense1.output)
    dense3.forward(activation1.output)
    loss = loss_activation.forward(dense3.output, y_train)
    if((epoch % 100) == 0):
            print(f'{epoch/100} %')
    # Backward pass
    loss_activation.backward(loss_activation.output, y_train)
    dense3.backward(loss_activation.dinputs)
    activation1.backward(dense3.dinputs)
    dense1.backward(activation1.dinputs)
    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense3)
    optimizer.post_update_params()


dense1.forward(X_test)
activation1.forward(dense1.output)
dense3.forward(activation1.output)
loss = loss_activation.forward(dense3.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)
y_test_argmax = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions==y_test_argmax)
print(f'accuracy:{accuracy}')
accuracy_ReLU.append(accuracy)