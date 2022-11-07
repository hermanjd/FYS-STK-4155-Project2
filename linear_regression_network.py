from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from backpropegation import *
import matplotlib.pyplot as plt

datapoints = 200

def f(x):
    return 4.2 + 3.2*x - 0.6*x**2

x = (np.random.rand(datapoints)*10).reshape(-1, 1)
print(x)
y = (f(x)).reshape(-1, 1)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(X_test)

dense1 = Layer_Dense(1, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 64)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(64, 1)
activation3 = Activation_Linear()
loss_function = Loss_MeanSquaredError()
optimizer = Optimizer_Adam(learning_rate=0.005, decay=1e-3)
accuracy_precision = np.std(y_train) / 250

for epoch in range(10001):

    # Perform a forward pass of our training data through this layer
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
    accuracy = np.mean(np.absolute(predictions - y_train) < accuracy_precision)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
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

mse = loss_function.calculate(y_test, activation3.output)

print(f'Neural network: {mse}, ')


poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(x)
predict_ = poly.fit_transform(y)

clf = linear_model.LinearRegression()
clf.fit(X_, vector)
print clf.predict(predict_)