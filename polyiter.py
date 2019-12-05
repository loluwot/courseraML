import matplotlib.pyplot as plt
import numpy as np
learning_rate = 0.08
n_degree = 35
dataset = [[1,1], [2,4], [3,9], [5,25], [6,36], [7,49], [8,64], [9,20], [10,100]]
x1,y1 = zip(*dataset)
dataset = [[point[0]**i for i in range(1, n_degree+1)] + [point[1]] for point in dataset]
epoch = 5000
new_dataset = [[] for i in range(len(dataset))]
*arr, last = zip(*dataset)
means = [0]
ranges = [1]
for l in arr:
	m = sum(l)/len(l)
	means.append(m)
	r = max(l)-min(l)
	ranges.append(r)
	for i in range(len(new_dataset)):
		new_dataset[i].append((l[i]-m)/r)
for i in range(len(last)):
	new_dataset[i].append(last[i])
print(new_dataset)
dataset = new_dataset
features = np.array([[1] + x[:-1] for x in dataset])
print(features)
coeff = np.array([0]*(n_degree+1));
def predict (feature, coeff):
	return np.dot(coeff, feature)
def transform(val):
	features = [(val**i-means[i])/ranges[i] for i in range(n_degree+1)]
	return features
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
for i in range(epoch):
	temp = []
	for n in range(len(dataset[0])):
		temp.append(coeff[n] - learning_rate*partialDer(dataset, coeff, n))
	coeff = temp;
	#print(coeff)
	print("Loss: " + str(loss(dataset, coeff)))

print(predict(transform(4), coeff))
pointX = np.linspace(min(x1),max(x1),100)
print(pointX)
func = predict(transform(pointX), coeff)
plt.plot(x1, y1, 'ro')
plt.plot(pointX, func)
plt.show()