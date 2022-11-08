import numpy as np
from matplotlib import cm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

#Function for making design matrix (This function comes from the code examples found in https://compphysics.github.io/MachineLearning/doc/pub/week35/html/week35.html)
def create_X(x, y, n ):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)
	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

#function for finding betavalues (Takes in a design matrix and the y values)
def findBetaValues(X,y):
	XT = X.transpose()
	core = np.matmul(XT, X)
	coreInversed = np.linalg.inv(core)
	XTY = np.matmul(coreInversed, XT)
	B = np.matmul(XTY,y)
	return B

#This function finds the values to the estimated function ( y = Xb ) (Takes in the designmatrix and the betavalues)
def findY(X,b):
	values = []
	for index_row, element_row in enumerate(X):
		row = 0.0;
		for index_column, element_column in enumerate(element_row):
			row = row + (element_column*b[index_column])
		values.append(row)
	return values

#Evaluate the mean
def evaluateMean(Y):
	sum = 0.0
	length = len(Y)
	for i in range(length):
		sum = sum + Y[i]
	return sum * (1/length)

#Evaluates the mean squared error
def evaluateMSE(Y,y):
	sum = 0.0
	length = len(Y)
	for i in range(length):
		sum = sum + pow((Y[i]-y[i]),2)
	return sum * (1/length)

#Evaluates the Rsquared 
def evaluateRSquared(Y,y):
	MSE = evaluateMSE(Y,y)
	mean = evaluateMean(Y)
	sum = 0.0
	length = len(Y)
	for i in range(length):
		sum = sum + pow((Y[i]-mean),2)
	return 1 - (MSE/sum)

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4

def FrankeFunctionWithNoise(x,y,noise):
    frank = FrankeFunction(x,y)
    return frank + np.random.normal(0, noise, frank.shape)

def BootstraFunction(data, datapoints):
	t = np.zeros(datapoints)
	n= len(data)
	for i in range (datapoints):
		t[i]=np.mean(data[np.random.randint(0,n,n)])
	# analysis    
	print(np.mean(data), np.std(data),np.mean(t),np.std(t))
	return t

def PolynomialOLSBootstrapResampling(x, y, z, testSize, polynomialDegrees, bootstrapDegree):
	x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x,y,z,test_size=testSize)
	polynomialDegrees += 1
	error = np.zeros(polynomialDegrees)
	bias = np.zeros(polynomialDegrees)
	variance = np.zeros(polynomialDegrees)
	polydegree = np.zeros(polynomialDegrees)
	for i in range(0, polynomialDegrees):
		z_pred = []
		for j in range(bootstrapDegree):
			x_, y_, z_ = resample(x_train, y_train, z_train)
			designMatrix = create_X(x_,y_,i)
			B = findBetaValues(designMatrix,z_)
			testMatrix = create_X(x_test,y_test,i)
			Y = findY(testMatrix,B)
			z_pred.append(np.asarray(Y))
		arr = np.asarray(z_test)
		arr2 = np.asarray(z_pred)
		polydegree[i] = i
		error[i] = np.mean( np.mean((arr - arr2)**2, axis=0, keepdims=True) )
		bias[i] = np.mean( (arr - np.mean(arr2, axis=0, keepdims=True))**2 )
		variance[i] = np.mean( np.var(arr2, axis=0, keepdims=True) )
	return polydegree, error, bias, variance

def PolynomialOLSCrossValidation(x, y, z, polynomialDegrees, k):
	polynomialDegrees += 1
	error = np.zeros(polynomialDegrees)
	polydegree = np.zeros(polynomialDegrees)
	x_split = np.array_split(x, k)
	y_split = np.array_split(y, k)
	z_split = np.array_split(z, k)

	for i in range(0, polynomialDegrees):
		z_pred = []
		for j in range(k):
			x_train = np.delete(x_split, j, axis=0).flatten()
			y_train = np.delete(y_split, j, axis=0).flatten()
			z_train = np.delete(z_split, j, axis=0).flatten()
			x_test = x_split[j]
			y_test = y_split[j]
			designMatrix = create_X(x_train,y_train,i)
			B = findBetaValues(designMatrix,z_train)
			testMatrix = create_X(x_test,y_test,i)
			Y = findY(testMatrix,B)
			z_pred.append(np.asarray(Y))
		
		arr = np.asarray(z)
		arr2 = np.asarray(z_pred).flatten()
		polydegree[i] = i
		error[i] = np.mean((arr - arr2)**2)
	
	return polydegree, error