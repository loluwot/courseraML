import matplotlib.pyplot as plt
import numpy as np
learning_rate = 0.01
dataset = [(1,2), (2,3), (3,5), (7, 10)] #array of tuples (x,y)
#linear function to attempt to model points of form a+bx
a = 0 
b = 0
def predict(a, b, x):
	return a + b*x
def loss(dataset, a, b):
	sum = 0
	for point in dataset:
		sum = sum + (predict(a, b, point[0])-point[1])**2
	return sum/(2*len(dataset))
def aPartialDer (dataset, a, b):
	sum = 0;
	for point in dataset:
		sum = sum + (predict(a, b, point[0])-point[1])
	return sum/len(dataset)
def bPartialDer (dataset, a, b):
	sum = 0;
	for point in dataset:
		sum = sum + (predict(a, b, point[0])-point[1])*point[0]
	return sum/len(dataset)
#gradient descent
while (abs(aPartialDer(dataset, a, b)) > 0.001) and (abs(bPartialDer(dataset, a, b)) > 0.001):
	temp1 = a - learning_rate*aPartialDer(dataset, a, b)
	temp2 = b - learning_rate*bPartialDer(dataset, a, b)
	a = temp1
	b = temp2
	print(str(a)+ " " + str(b))
	print("Loss: " + str(loss(dataset, a, b)))
x, y = zip(*dataset)
pointX = np.linspace(-10,10,100)
line = a + b*pointX
plt.plot(x, y, 'ro')
plt.plot(pointX, line)
plt.show()