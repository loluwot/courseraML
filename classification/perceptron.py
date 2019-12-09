import numpy as np
import math
dataset = [[1,1,[1]], [1,0,[0]], [0,1,[0]]]
layers = [np.matrix([[1,1,1], [1,1,1]]),np.matrix([1,1,1])]
#reg = 100
def activation(z):
	return np.matrix(1/(1+math.e**(-1*z))[0])
def predict(matrix, inputs):
	return activation(np.array(np.matmul(matrix, inputs)))
def loss (layers, dataset):
	sum = 0
	for feature in dataset:
		output = forward_prop([1]+feature[:-1], layers)[-1]
		for i in range(len(feature[-1])):
			sum += feature[-1][i]*math.log(output[i+1])+(1-feature[-1][i])*math.log(1-output[i+1])
	return -1*sum/len(dataset)		
		
def forward_prop(inputs, layers):
	last_layer = inputs
	a = [last_layer]
	for layer in layers:
		last_layer = np.append([1], predict(layer, last_layer))
		a.append(last_layer)
	return a

def back_prop(layers, dataset):	
	true_delta = [0]*len(layers)
	for feature in dataset:
		output = forward_prop([1]+feature[:-1], layers)
		delta = [np.transpose(np.matrix(np.append([1],output[-1][1:] - feature[-1])))]
		#print(np.delete(delta[0], 0))
		for i in range(len(layers)-1,-1,-1):
			print(np.transpose(layers[i]).shape)
			#print(delta[0].shape)
			print(np.delete(delta[0], 0,0).shape)
			print(str(np.matmul(np.transpose(layers[i]),np.delete(delta[0], 0,0))) + str(i))
			#print(np.transpose(np.matrix(output[i])))
			print(np.multiply(np.multiply(np.matmul(np.transpose(layers[i]),np.delete(delta[0], 0,0)),np.transpose(np.matrix(output[i]))), np.transpose(np.matrix(1-np.array(output[i])))))
			delta.insert(0, np.multiply(np.multiply(np.matmul(np.transpose(layers[i]),np.delete(delta[0], 0,0)),np.transpose(np.matrix(output[i]))), np.transpose(np.matrix(1-np.array(output[i])))))
		
		print(delta)
		
print(back_prop(layers, dataset))