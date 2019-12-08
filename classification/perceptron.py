import numpy as np
import math
inputs = [1]+[1,1]
layers = [np.matrix([[-30, 20, 20], [10, -20, -20]]),np.matrix([-10, 20, 20])]
def activation(z):
	return np.matrix(1/(1+math.e**(-1*z))[0])
def predict(matrix, inputs):
	return activation(np.array(np.matmul(matrix, inputs)))

last_layer = inputs
for layer in layers:
	last_layer = np.append([1], predict(layer, last_layer))
	print(last_layer)
print(last_layer[1] > 0.5)
