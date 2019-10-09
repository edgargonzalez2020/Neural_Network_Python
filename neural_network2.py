import math
import sys
import numpy as np


class Layer:
	def __init__(self, input_shape, output_shape, learning_rate = 1.0):
		self.input_shape = int(input_shape)
		self.output_shape = int(output_shape)
		self.lr = int(learning_rate)
		self.weights = np.random.uniform(-0.5, 0.5, (self.input_shape, self.output_shape))
		self.biases = np.ones(self.output_shape)
	def update_lr(self, x):
		self.lr = x
	def feedforward(self, x):
		return self.sigmoid(np.dot(x, self.weights) + self.biases)
	@staticmethod
	def sigmoid(x):
		return 1 / (1 - np.exp(-x))
	@staticmethod
	def sigmoid_prime(x):
		return x * (1 - x)


class LossFunction:
	def __init__(self):
		pass
	@staticmethod
	def derivative(actual, predicted):
		return predicted - actual
	@staticmethod
	def delta(actual, predicted,activation_function_derivative):
		return (predicted - actual) * activation_function_derivative(predicted)




class NN:
	def __init__(self, training_path, test_path, layer_count, units_per_layer, rounds):
		self.input_shape = None
		self.output_shape = None
		self.training_max = None
		self.test_max = None

		self.layers = int(layer_count)
		self.units_per_layer = int(units_per_layer)
		self.epochs = int(rounds)


		self.training_data = self.load(training_path, True) / self.training_max
		self.test_data = self.load(test_path, False) / self.test_max
		self.network = []
		for j in range(self.layers):
			if j == 0: 
				self.network.append(Layer(self.input_shape, self.units_per_layer))
				continue
			self.network.append(Layer(self.units_per_layer, self.units_per_layer))
		self.network.append(Layer(self.units_per_layer, self.output_shape))
	def load(self, path, is_training):
		data = np.loadtxt(path)
		if is_training:
			self.make_one_hot(data[: , [-1]], True)
			self.training_max = np.amax(data)
		else:
			self.make_one_hot(data[:, [-1]], False)
			self.test_max = np.amax(data)
		self.input_shape = data.shape[1] - 1
		self.output_shape = self.training_labels.shape[1]
		return data[: , [x for x in range(data.shape[1] - 1 )]]
	def make_one_hot(self, labels, is_training):
		col_max = np.asscalar(np.max(labels))
		ones_arr = np.zeros((labels.shape[0], int(col_max)))
		for i,x in enumerate(labels):
			ones_arr[i][max(0, int(x) - 1)] = 1
		if is_training:
			self.training_labels = ones_arr
		else:
			self.test_labels = ones_arr
	def feedforward(self, x):
		Z = []
		input_ = x
		for y in self.network:
			Z.append(y.feedforward(input_))
			input_ = Z[-1]
		return Z
	def backprop(self, Z,label):
		## The indices in this loop are iffy make sure to check back on these if a 
		## problem arises
		derivative_cache = {}
		delta = LossFunction.delta(label, Z[-1], Layer.sigmoid_prime)
		derivative = np.dot(np.array(Z[:-2]).T, delta)
		derivative_cache[len(self.network) - 1] = (derivative, delta)
		for i in range(len(Z) - 1, 1, -1):
			delta = np.dot(delta, self.network[i].weights.T)
			derivative = np.dot(np.array(Z[: i - 1]).T, delta)
			derivative_cache[i - 1] = (derivative, delta)
		for k,v in derivative_cache.items():
			self.network[k].weights -= self.network[k].lr * v[0]
			self.network[k].biases -= self.network[k].lr * np.mean(v[1], 0)
			self.network[k].update_lr(self.network[k].lr * 0.98)

	def train(self):
		for j in range(self.epochs):
			for i, x in enumerate(self.training_data):
				Z = self.feedforward(x)
				self.backprop(Z,self.training_labels[i].reshape(1,self.training_labels.shape[1]))
	def predict(self, x):
		return self.feedforward(x)[-1]
def main():
	if len(sys.argv) < 6:
		print('Usage: [path to training file] [path to test file] layer_count units_per_layer rounds')
	classifier = NN(*sys.argv[1:6])
	classifier.train()
	print('Actual:', classifier.test_labels[0])
	print('Predicted:', classifier.predict(classifier.test_data[0]))
if __name__ == '__main__':
	main()