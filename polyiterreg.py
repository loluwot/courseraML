import matplotlib.pyplot as plt
import numpy as np
learning_rate = 0.175
n_degree = 5
approx_degree = 3 #approximate degree of data
reg_constant = 100
dataset = [[1,1], [2,8], [3,27], [5,125], [6,6**3], [7,7**3], [8,8**3], [9,9**3], [10,10**3]]
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
		min = min and (abs(partialDer(dataset, coeff, n)) < 0.005)
	return min		
#gradient descent
while not isMin(dataset, coeff):
	temp = []
	for n in range(len(dataset[0])):
		if (n <= approx_degree):
			temp.append(coeff[n] - learning_rate*partialDer(dataset, coeff, n))
		else:
			temp.append(coeff[n]*(1-reg_constant*learning_rate/len(dataset)) - learning_rate*partialDer(dataset, coeff, n))
	coeff = temp
	#print(coeff)
	print("Loss: " + str(loss(dataset, coeff)))

print(predict(transform(11), coeff))
pointX = np.linspace(min(x1),max(x1)+5,100)
print(pointX)
func = predict(transform(pointX), coeff)
plt.plot(x1, y1, 'ro')
plt.plot(pointX, func)
plt.show()