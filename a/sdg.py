from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from backpropegation import *
import matplotlib.pyplot as plt

n = 100

def f(x):
    return 4.2 + 3.2*x

x = np.random.rand(n)*10
y = f(x)

beta = np.random.rand(3)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

def design_matrix(x,M):
    desm = np.zeros(shape =(len(x), M+1))
    for i in range(0, M+1):
        desm[:,i] = np.power(x, i).reshape(x.shape[0],)
    return desm


X = design_matrix(X_train,2)
# Hessian matrix
H = (2.0/n)* X.T @ X


print((X.T @ X) @ (X.T @ y))
# Get the eigenvalues
EigValues, EigVectors = np.linalg.eig(H)
print(f"Eigenvalues of Hessian Matrix:{EigValues}")

beta_linreg = np.linalg.inv(X.T @ X) @ (X.T @ y)
print(beta_linreg)
beta = np.random.randn(2,1)

eta = 1.0/np.max(EigValues)
Niterations = 1000

for iter in range(Niterations):
    gradient = (2.0/n)*X.T @ (X @ beta-y)
    beta -= eta*gradient

print(beta)
xnew = np.array([[0],[2]])
xbnew = np.c_[np.ones((2,1)), xnew]
ypredict = xbnew.dot(beta)
ypredict2 = xbnew.dot(beta_linreg)
plt.plot(xnew, ypredict, "r-")
plt.plot(xnew, ypredict2, "b-")
plt.plot(x, y ,'ro')
plt.axis([0,2.0,0, 15.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Gradient descent example')
plt.show()