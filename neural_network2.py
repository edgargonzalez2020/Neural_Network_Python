import math
import sys
import numpy as np


class Layer:
	def __init__(self, input_shape, output_shape, learning_rate = 1.0):
		self.input_shape = int(input_shape)
		self.output_shape = int(output_shape)
		self.lr = int(learning_rate)
		np.random.seed(1)
		self.weights = np.random.uniform(-0.5, 0.5, (self.input_shape, self.output_shape))
		self.biases = np.zeros(self.output_shape)
	def update_lr(self, x):
		self.lr = x
	def feedforward(self, x):
		return self.sigmoid(np.dot(x, self.weights))
	def sigmoid(self,x):
		return 1 / (1 + np.exp(-x))
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
	def delta(actual, predicted):
		return (predicted - actual) * (predicted * (1 - predicted))
	@staticmethod
	def loss(actual,predicted):
		return np.mean((predicted-actual)**2)




class NN:
	def __init__(self, training_path, test_path, layer_count, units_per_layer, rounds):
		self.input_shape = None
		self.output_shape = None
		self.training_max = None

		self.mapping = {}

		self.layers = int(layer_count)
		self.units_per_layer = int(units_per_layer)
		self.epochs = int(rounds)


		self.training_data = self.load(training_path, True) / self.training_max
		self.test_data = self.load(test_path, False) / self.training_max
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
		Z = []
		input_ = x
		for i,y in enumerate(self.network):
			res = y.feedforward(input_)
			Z.append(res)
			input_ = Z[-1]
		return Z
	def backprop(self, Z,label):
		derivative_cache = {}
		## Delta value of the output layer 
		delta = LossFunction.delta(label, Z[-1])
		#derivative = np.dot(np.array(Z[-2]).reshape(1,len(Z[-2])).T, delta)
		derivative_cache[len(self.network) - 1] = delta
		for i in reversed(range(1,len(Z))):
			delta = np.dot(delta, self.network[i].weights.T) * Layer.sigmoid_prime(Z[i - 1])
			#derivative = np.dot( np.array(Z[i - 1]).reshape(1,len(Z[i-1])).T, delta)
			derivative_cache[i - 1] = delta
		for k,v in derivative_cache.items():
			self.network[k].weights = self.network[k].weights - (self.network[k].lr * v)
			#self.network[k].biases = self.network[k].biases - self.network[k].lr * np.mean(v[1], 0)
			self.network[k].update_lr(self.network[k].lr * 0.98)
	def train(self):
		for j in range(self.epochs):
			for i, x in enumerate(self.training_data):
				Z = self.feedforward(x)
				self.backprop(Z,self.training_labels[i].reshape(1,self.training_labels.shape[1]))
	def predict(self, x):
		return self.feedforward(x)[-1]
	def run_predictions(self):
		count = 1
		avg = 0
		for i,y in enumerate(self.test_data):
			prediction = self.feedforward(y)[-1]
			print(prediction)
			predicted = self.get_class(prediction)
			true = self.get_class(self.test_labels[i])
			accuracy = 1 if predicted == true else 0
			print("ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n" % (
				count, predicted, true, accuracy))
			count += 1
			avg += accuracy
		print("classification accuracy=%6.4f" % (avg / (count-1)))

	def get_class(self, vec):
		return np.argmax(vec, axis=0)
	def print_meta(self):
		print(f'Layers: {self.layers}')
		print(f'Units: {self.units_per_layer}')
		print(f'Epochs: {self.epochs}')
		print(f'Input Shape: {self.input_shape}')
		print(f'Output Shape: {self.output_shape}')
def main():
	if len(sys.argv) < 6:
		print('Usage: [path to training file] [path to test file] layer_count units_per_layer rounds')
	classifier = NN(*sys.argv[1:6])
	#classifier.print_meta()
	classifier.train()
	classifier.run_predictions()
	# print()
	# print()
	# print()
	# for x,i in enumerate(classifier.test_data):
	# 	prediction = classifier.feedforward(classifier.test_data[x])[-1]
	# 	print(prediction, prediction.shape)
	# 	print(classifier.test_labels[x], classifier.test_labels[x].shape)
	# 	print(classifier.get_class(prediction))
	# 	print()
	# 	print()
if __name__ == '__main__':
	main()