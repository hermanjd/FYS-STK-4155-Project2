import sys
sys.path.append("..")
import numpy as np
from sklearn.model_selection import train_test_split
from NeuralNetFunctions import *
import matplotlib.pyplot as plt
#input file

learning_rate = 0.005

data = np.loadtxt(".././formatted_wdbc.data",delimiter=',')
X = data[:,2:]
y = data[:, 1].astype(int)
new_matrix = X / X.max(axis=0)
X_train, X_test, y_train, y_test = train_test_split(new_matrix, y, test_size=0.2, random_state=42)

y_train = y_train.reshape(-1, 1)

dense1 = Layer_Dense(X.shape[1], 32)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(32, 1)
activation2 = Activation_Sigmoid()
loss_function = Loss_BinaryCrossentropy()
optimizer = Optimizer_Adam(learning_rate=learning_rate, decay=5e-5)

batch_size = 32
epochs = 70

loss_list = []

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
        data_loss = loss_function.calculate(activation2.output, y_batch)
        loss = data_loss 
        predictions = (activation2.output > 0.5) * 1
        accuracy = np.mean(predictions==y_batch)
        loss_list.append(data_loss)
        if not epoch % 100:
                print(f'epoch: {epoch}, ' +
                    f'acc: {accuracy:.3f}, ' +
                    f'loss: {loss:.3f}, ' +
                    f'lr: {optimizer.current_learning_rate}')

        # Backward pass
        loss_function.backward(activation2.output, y_batch)
        activation2.backward(loss_function.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update weights and biases
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()


y_test = y_test.reshape(-1, 1)

dense1.forward(X_test)

activation1.forward(dense1.output)

dense2.forward(activation1.output)

activation2.forward(dense2.output)

loss = loss_function.calculate(activation2.output, y_test)

MSE = np.square(np.subtract(y_test,activation2.output)).mean()
predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(activation2.output.round()==y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}, MSE: {MSE:.3f}')


plt.plot(loss_list)
plt.show()