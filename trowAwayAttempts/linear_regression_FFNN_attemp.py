import sys
sys.path.append("..")
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from NeuralNetFunctions import *
import matplotlib.pyplot as plt

datapoints = 1000
learning_rate = 0.001
momentum=0.9

def f(x):
    return 4.2 + 3.2*x - 0.6*x**2

x = ((np.random.rand(datapoints)*15)-5).reshape(-1, 1)
y = (f(x)).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

dense1 = Layer_Dense(1, 16)
activation1 = Activation_Sigmoid()
dense2 = Layer_Dense(16, 16)
activation2 = Activation_Sigmoid()
dense3 = Layer_Dense(16, 1)
activation3 = Activation_Linear()
loss_function = Loss_MeanSquaredError()
optimizer = Optimizer_SGD(learning_rate=learning_rate, momentum=momentum)

loss_list = []
true_loss = []


for epoch in range(20000):
    dense1.forward(X_train)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    data_loss = loss_function.calculate(activation3.output, y_train)
    regularization_loss = \
        loss_function.regularization_loss(dense1) + \
        loss_function.regularization_loss(dense2) + \
        loss_function.regularization_loss(dense3)

    loss = data_loss + regularization_loss
    predictions = activation3.output
    loss_list.append(data_loss)
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')

    # Backward pass
    loss_function.backward(activation3.output, y_train)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
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
activation3.forward(dense3.output)

mse = loss_function.calculate(activation3.output, y_test)

print(f'Neural network: {mse}, ')

plt.plot(loss_list)
plt.show()
