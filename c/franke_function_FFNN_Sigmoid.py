import sys
sys.path.append("..")
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from backpropegation import *
import matplotlib.pyplot as plt

datapoints = 100
learning_rate = 0.01
momentum=0.9

def f(x):
    return 4.2 + 3.2*x - 0.6*x**2


def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4

def FrankeFunctionWithNoise(x,y,noise):
    frank = FrankeFunction(x,y)
    return frank + np.random.normal(0, noise, frank.shape)

N = 1000
x = np.random.uniform(0, 1, N)
y = np.random.uniform(0, 1, N)
z = (FrankeFunction(x,y)).reshape(-1, 1) #adding some noise to the data
XY = np.vstack((x, y)).T

XY_train, XY_test, z_train, z_test = train_test_split(XY,z,test_size=0.2)

dense1 = Layer_Dense(2, 750)
activation1 = Activation_Sigmoid()
dense2 = Layer_Dense(750, 1)
activation2 = Activation_Linear()
loss_function = Loss_MeanSquaredError()
optimizer = Optimizer_SGD(learning_rate=learning_rate, momentum=momentum)

loss_list = []

batch_size = 32
epochs = 100
for epoch in range(epochs):
    print(epoch)
    for batch in range(int(len(XY_train)/batch_size)):
        xy_batch = XY_train[batch*batch_size:(batch+1)*batch_size]
        y_batch = z_train[batch*batch_size:(batch+1)*batch_size]
        dense1.forward(xy_batch)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        data_loss = loss_function.calculate(activation2.output, y_batch)

        loss_list.append(data_loss)
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


dense1.forward(XY_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

mse = loss_function.calculate(activation2.output, z_test)

print(f'Neural network: {mse}, ')

plt.plot(loss_list)
plt.show()