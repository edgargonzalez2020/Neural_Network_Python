import sys
import math
import numpy as np 
from scipy.stats import truncnorm
def get_truncated_normal(mean=0, sd=1, low=-0.5, upp=0.5):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
def sigmoid(x):
	return 1 / (1 + np.exp(-x))
def sigmoid_prime(x):
	return sigmoid(x) * (1 - sigmoid(x))
def main():
	if len(sys.argv) < 6:
		print('Usage: [path to training file] [path to test file] layer_count units_per_layer rounds')
	classifier = NeuralNet(*sys.argv[1:6])
	classifier.train()
	classifier.run_predictions()


class Layer:
	def __init__(self, input_dim, output_dim):
		self.lr = 1
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.weights = np.random.uniform(-0.5,0.5, (self.input_dim, self.output_dim))
		self.biases = np.zeros((output_dim,1))
	def feedforward(self, x):
		return sigmoid(np.dot(self.weights.T, x))
	def update_lr(self):
		self.lr *= .98


class NeuralNet:
	def __init__(self, train_path, test_path, layers, units_per_layer, epochs):
		self.layer_count = int(layers)
		self.units = int(units_per_layer)
		self.epochs = int(epochs)

		self.mapping = {}

		self.training_data = self.load(train_path, True) / self.training_max
		self.test_data = self.load(test_path, False) / self.training_max


		self.network = self.load_network()
	def get_class(self, vec):
		return np.argmax(vec, axis=0)
	def load_network(self):
		net = []
		if self.layer_count == 0:
			net.append(Layer(self.input_shape,self.output_shape))
			return net
		for x in range(self.layer_count):
			if x == 0:
				net.append(Layer(self.input_shape, self.units))
				continue
			net.append(Layer(self.units, self.units))
		net.append(Layer(self.units, self.output_shape))
		return net
	def load(self, path, is_training):
		data = np.loadtxt(path)
		if is_training:
			self.make_one_hot(data[: , [-1]], True)
			self.training_max = np.amax(data)
		else:
			self.make_one_hot(data[:, [-1]], False)
		self.input_shape = data.shape[1] - 1
		self.output_shape = self.training_labels.shape[1]
		return data[: , [x for x in range(data.shape[1] - 1 )]]
	def make_one_hot(self, labels, is_training):
		labels_set = set()
		for x in labels:
			labels_set.add(int(np.asscalar(x)))
		labels_set = sorted(labels_set)
		for i,x in enumerate(labels_set):
			self.mapping[x] = i
		zeros_arr = np.zeros((labels.shape[0],len(labels_set)))
		for i,y in enumerate(labels):
			number = int(np.asscalar(y))
			idx = self.mapping[number]
			zeros_arr[i][idx] = 1
		if is_training:
			self.training_labels = zeros_arr
		else:
			self.test_labels = zeros_arr
	def feedforward(self, x):
		input_ = x
		Z = []
		if self.layer_count == 0:
			return self.network[0].feedforward(input_)
		else:
			for i,y in enumerate(self.network):
				Z.append(y.feedforward(input_))
				input_ = Z[-1]
		return Z
	def backprop(self, Z, label):
		if self.layer_count == 0:
			pass
		elif self.layer_count == 1:
			pass
		else:
			pass		
			# hm = {}
			# output_layer = Z[-1]
			# error = output_layer - label.reshape(len(label),1)
			# sigmoid_term = sigmoid_prime(Z[-1])
			# delta = error * sigmoid_term
			# dw = np.dot(Z[-2],delta.T)

			# hm[len(Z) - 1] = dw,delta
			# for y in reversed(range(2, len(self.network))):
			# 	delta = np.dot(self.network[y].weights, delta) * sigmoid_prime(Z[y-1]) * Z[y-1]
			# 	# dw = np.dot(Z[y-1],delta.T)
			# 	hm[y-1] = delta,dw
			# for k,v in hm.items():
			# 	self.network[k].weights = self.network[k].weights - (self.network[k].lr * v[0])
			# 	self.network[k].update_lr()
			# 	# self.network[k].biases -= self.network[k].lr * np.mean(v[0], 0)
	def train(self):
		iterations = 0
		for x in range(self.epochs):
			for i,y in enumerate(self.training_data):
				Z = self.feedforward(y.reshape(len(y),1))
				self.backprop(Z, self.training_labels[i])
	def run_predictions(self):
		count = 1
		avg = 0
		for i,y in enumerate(self.test_data):
			prediction = self.feedforward(y.reshape(len(y),1))[-1]
			predicted = self.get_class(prediction)
			true = self.get_class(self.test_labels[i])
			accuracy = 1 if predicted == true else 0
			print("ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n" % (count, predicted, true, accuracy))
			count += 1
			avg += accuracy
		print("classification accuracy=%6.4f" % (avg / (count-1)))
	def single_step():
		x = self.test_data[0]
		x_label = self.test_labels[0]
		Z = self.feedforward(x)
		print('Z:', Z)
		print()
		print('Output:', Z[-1])
		print()
if __name__ == '__main__':
	main()