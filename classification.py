import numpy as np
import math
dataset = [[1,2,1], [2,3,0], [3,4,0]]
learning_rate = 0.2
features = np.array([[1] + x[:-1] for x in dataset])
coeff = np.array([0]*len(dataset[0]))

def predict(features, coeff):
	val = np.dot(features, coeff)
	return 1/(1+math.e**(-1*val)) #logistic function

def loss(dataset, coeff):
	sum = 0
	for feature in dataset:
		sum += feature[-1]*math.log(predict([1]+feature[:-1], coeff))+(1-feature[-1])*math.log(1-predict([1]+feature[:-1], coeff))
	return -1*sum/len(dataset)

def partialDer(dataset, coeff, n):
	sum = 0
	for feature in dataset:
		sum += (predict([1]+feature[:-1], coeff)-feature[-1])*([1]+feature[:-1])[n]
	return sum/(len(dataset))

def isMin(dataset, coeff):
	ismin = True
	for i in range(len(dataset[0])):
		ismin = ismin and (abs(partialDer(dataset, coeff, i)) < 0.001)
	return ismin

while not isMin(dataset, coeff):
	temp = []
	for n in range(len(dataset[0])):
		temp.append(coeff[n] - learning_rate*partialDer(dataset, coeff, n))
	coeff = temp;
	print(coeff)
	print("Loss: " + str(loss(dataset, coeff)))

print(predict([1,0,1], coeff))