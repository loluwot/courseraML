import numpy as np
import matplotlib.pyplot as plt
dataset = [[1,2,3], [2,3,4], [4,5,6]]
features = np.matrix([[1] + feature[:-1] for feature in dataset])
y = [feature[-1] for feature in dataset]
coeff = np.dot(np.matmul(np.linalg.pinv(np.matmul(features.transpose(),features)), features.transpose()), y) #normal equation to solve partial derivatives at 0, automatically finds local minima

def predict (feature, coeff):
	return np.dot(coeff, feature)

print(predict([1,4,5], coeff))
