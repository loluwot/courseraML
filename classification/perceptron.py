import numpy as np
import math
learning_rate = 0.5
dataset = [[1,1,[1]], [1,0,[0]], [0,1,[0]], [0,0,[1]]]
layers = [np.matrix(np.random.rand(2,3)),np.matrix(np.random.rand(1,3))]
print(layers[0])
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
		#print(layer.shape)
		#print(np.matrix(last_layer).shape)
		#print(predict(layer, last_layer).tolist())
		last_layer = np.append([1], predict(layer, last_layer).tolist())
		#print(last_layer)
		a.append(last_layer)
	return a

def back_prop(layers, dataset):	
	true_delta = [0]*len(layers)
	for feature in dataset:
		output = forward_prop([1]+feature[:-1], layers)
		#print(output)
		delta = [np.transpose(np.matrix(np.append([1],output[-1][1:] - feature[-1])))]
		#print(np.delete(delta[0], 0))
		for i in range(len(layers)-1,0,-1):
			#print(np.transpose(layers[i]).shape)
			#print(delta[0].shape)
			#print(np.delete(delta[0], 0,0).shape)
			#print(str(np.matmul(np.transpose(layers[i]),np.delete(delta[0], 0,0))) + str(i))
			#print(np.transpose(np.matrix(output[i])))
			#print(np.multiply(np.multiply(np.matmul(np.transpose(layers[i]),np.delete(delta[0], 0,0)),np.transpose(np.matrix(output[i]))), np.transpose(np.matrix(1-np.array(output[i])))))
			delta.insert(0, np.multiply(np.multiply(np.matmul(np.transpose(layers[i]),np.delete(delta[0], 0,0)),np.transpose(np.matrix(output[i]))), np.transpose(np.matrix(1-np.array(output[i])))))
		
		for i in range(len(layers)):
			#print(delta[i][1:].shape)
			#print(np.matrix(output[i]).transpose().shape)
			true_delta[i] = true_delta[i] + np.matmul(delta[i][1:], np.matrix(output[i]))
		#print(delta)
		# print("-----")
		# print(true_delta)
		# print("-----")
	return true_delta
for i in range(1000):	
	print("-------")
	print(loss(layers, dataset))
	print("-------")
	t_delta = back_prop(layers, dataset)
	for i in range(len(t_delta)):
		layers[i] -= t_delta[i]*learning_rate/len(dataset)
print(forward_prop([1,0,0], layers))