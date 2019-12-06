import numpy as np
import math
import matplotlib.pyplot as plt
n_degree = 2
dataset = [[1,1], [2,0], [3,0], [4,1]]
x1,y1 = zip(*dataset)
learning_rate = 0.2
features = np.array([[1] + x[:-1] for x in dataset])
dataset = [[point[0]**i for i in range(1, n_degree+1)] + [point[1]] for point in dataset]

coeff = np.array([0]*len(dataset[0]))

def predict(features, coeff):
	val = np.dot(coeff, features)
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

def transform (x):
	arr = [x**i for i in range(n_degree+1)]
	return arr

print(predict(transform(5), coeff))
pointX = np.linspace(min(x1),max(x1),100)
print(pointX)
func = predict(transform(pointX), coeff)
plt.plot(x1, y1, 'ro')
plt.plot(pointX, func)
plt.show()