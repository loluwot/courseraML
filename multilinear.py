import matplotlib.pyplot as plt
import numpy as np
learning_rate = 0.01
n_features = 2 
dataset = [[1,2,3], [2,3,4], [4,5,6]]
features = np.array([[1] + x[:-1] for x in dataset])
print(features)
coeff = np.array([0]*(n_features+1));
def predict (feature, coeff):
	return np.dot(coeff, feature)
def loss (dataset, coeff):
	sum = 0
	for feature in dataset:
		sum = sum + (predict([1]+feature[:-1], coeff)-feature[-1])**2
	return sum/(2*len(dataset))
#partial derivative of loss function for nth feature
def partialDer(dataset, coeff, n):
	sum = 0
	for feature in dataset:
		sum = sum+(predict([1]+feature[:-1], coeff)-feature[-1])*([1]+feature[:-1])[n]
	return sum/(len(dataset))
#checks if at local minimum
def isMin(dataset, coeff):
	min = True
	for n in range(len(dataset[0])):
		min = min and (abs(partialDer(dataset, coeff, n)) < 0.001)
	return min		
#gradient descent
while (not isMin(dataset, coeff)):
	temp = []
	for n in range(len(dataset[0])):
		temp.append(coeff[n] - learning_rate*partialDer(dataset, coeff, n))
	coeff = temp;
	print(coeff)
	print("Loss: " + str(loss(dataset, coeff)))