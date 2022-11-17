import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from backpropegation import *
import gzip
import matplotlib.pyplot as plt

num_images = 55000
image_size = 28
neurons = 256
batch_size = 32
epochs = 2
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

accuracy_train = []
accuracy_test = []

dense1 = Layer_Dense(X_train.shape[1], neurons, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(neurons, neurons,  weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(neurons, 10)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(decay=decay, learning_rate=learning_rate, momentum=momentum)


for epoch in range(epochs):
    print(epoch)
    for batch in range(int(len(X_train)/batch_size)):
        x_batch = X_train[batch*batch_size:(batch+1)*batch_size]
        y_batch = y_train[batch*batch_size:(batch+1)*batch_size]
        if(batch % batch_size == 0):
            print(batch)
        dense1.forward(x_batch)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        dense3.forward(activation2.output)
        data_loss = loss_activation.forward(dense3.output, y_batch)

        regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)

        predictions = np.argmax(loss_activation.output, axis=1)
        y_test_argmax = np.argmax(y_batch, axis=1)
        accuracy = np.mean(predictions==y_test_argmax)
        accuracy_train.append(accuracy)
        # Calculate overall loss
        loss = data_loss + regularization_loss
        # Backward pass
        #print("loss_activation.output")
        #print(loss_activation.output)
        loss_activation.backward(loss_activation.output, y_batch)
        #print("loss_activation.dinputs")
        #print(loss_activation.dinputs)
        dense3.backward(loss_activation.dinputs)
        #print("dense3.dinputs")
        #print(dense3.dinputs)
        activation2.backward(dense3.dinputs)
        #print("activation2.dinputs")
        #print(activation2.dinputs)
        dense2.backward(activation2.dinputs)
        #print("dense2.dinputs")
        #print(dense2.dinputs)
        activation1.backward(dense2.dinputs)
        #print("dense3.dinputs")
        #print(activation1.dinputs)
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
        accuracy_test.append(np.mean(predictions==y_test_argmax))



print(accuracy_train[-1])
print(accuracy_test[-1])
txt = f's_{num_images}_n_{neurons}_l_2_bs_{batch_size}_SGDS_(d=5e-7_lr=0.02_m=0.9)'
print(txt)
plt.title(txt)
plt.plot(accuracy_train)
plt.plot(accuracy_test)
plt.show()

#Might be good to check training and testing accuracy to see if the model is overfit

# Test results 80 Epoch, Optimizer_Adam(decay=5e-7, learning_rate=0.01)
# Training data: 0.97625 - Test Data: 0.9608

# Test results 100 Epoch, Optimizer_Adam(decay=5e-7, learning_rate=0.01)
# Training data: 0.9895 - Test Data: 0.9681

# Test results 500 Epoch, Optimizer_Adam(decay=5e-7, learning_rate=0.008)
# Training data: 0.9974 - Test Data:0.9715

# Note, while the test data accuracy seems to be going up with more epoch, we can very clearly see that the model is also overfitting to the training data due to the 
# high fit of the training set at 99.74% and the testing accuracy at 97.15%. This gives ut a 2.59% differance between training and test accuracy.

