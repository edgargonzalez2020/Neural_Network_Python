import numpy as np
import pandas
import sys
import operator
import math
class NN:
	def __init__(self, training_path, test_path, layer_count, units_per_layer, epochs):

		self.classes_train = {}
		self.idx_mapping = {}
		self.count = 0
		## Network hyper paramaters 
		self.layers = int(layer_count)
		self.units_per_layer = int(units_per_layer)
		self.epochs = int(epochs)
		self.U = self.units_per_layer * self.layers


		# Variables related to the data 
		self.input_shape = None
		self.output_shape = None
		self.load_classes(training_path, True)
		self.load_classes(test_path, False)
		self.training_path = training_path
		self.test_path = test_path
		self.train_labels = None
		self.test_labels = None
		self.test_data = None


		## Network architecture varaibles
		self.weights = self.create_weights()
	def sigmoid(self, x):
		return 1 / (1 + math.exp(-x))
	def sigmoid_der(self, z, expected):
		return (z - expected) * z * (1 - z)
	def load_data(self, data_path):
		return np.loadtxt(data_path)
	def load_classes(self, data_path, is_training):
		data = self.load_data(data_path)
		labels = self.make_one_hot(data[ :, [-1]])
		if is_training:
			for l, row in enumerate(data):
				self.count += 1
				class_label = tuple(labels[l])
				if class_label not in self.classes_train:
					self.classes_train[class_label] = np.hstack((1,row[:-1])), 
				else:
					self.classes_train[class_label] = np.vstack([self.classes_train[class_label][0], np.hstack((1,row[:-1]))])
			
			self.input_shape = self.classes_train[class_label].shape[1]
			self.D = self.classes_train[class_label].shape[1] 
			self.U += self.classes_train[class_label].shape[1]
			self.classes_train = dict(sorted(self.classes_train.items()))
			for i,x in enumerate(self.classes_train):
				self.idx_mapping[i] = x
			self.output_shape = len(self.idx_mapping.keys())
			self.U += self.output_shape
		else:
			ones_arr = np.ones((data.shape[0], 1))
			self.test_labels = labels
			self.test_data = data[:, [x for x in range(data.shape[1] - 1)]]
			self.test_data = np.hstack((ones_arr, self.test_data))
		data = self.load_data(data_path)
	def create_weights(self):
		weights = []
		if self.layers > 1:
			weights.append(np.random.uniform(-0.5, 0.5, (self.input_shape) * (self.units_per_layer + 1)))
			for i in range(self.layers):
				weights.append(np.random.uniform(-0.5, 0.5, (self.units_per_layer + 1) ** 2))
			weights.append(np.random.uniform(-0.5, 0.5, (self.units_per_layer + 1) * self.output_shape))
		else:
			weights.append(np.random.uniform(-0.5, 0.5, self.input_shape * self.output_shape))
		return weights
	def create_ZA(self, x):
		Z = np.zeros(self.U)
		A = np.zeros(self.U)
		for i in range(self.D):
			Z[i] = x[i]
		return Z,A
	def make_one_hot(self, labels):
		col_max = np.asscalar(np.max(labels))
		ones_arr = np.zeros((labels.shape[0], int(col_max)))
		for i,x in enumerate(labels):
			ones_arr[i][max(0, int(x) - 1)] = 1
		return ones_arr
	def feed_forward(self, x):
		## Not sure if this weights update will work
		Z, A = self.create_ZA(x)
		for x in self.weights:
			print(x, x.shape)
			print()
		for i in range(self.layers):
			break
		return Z,A
	def backprop(self, Z , A, t):
		idx = self.D + (self.units_per_layer * self.layers)
		Delta = np.zeros(self.U)
		# for i, x in enumerate(range(idx, len(Z))):
		# 	print(Z[x], t[i], self.sigmoid_der(Z[x], t[i]))
	def train(self):
		for i in self.classes_train:
			for x in self.classes_train[i]:
				Z,A = self.feed_forward(x)
				self.backprop(Z,A, i)
				break
			break
	def predict(self, x):
		pass
	def get_one_hot(self, x, class_label, count, avg):
		pass
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
	classifier.train()
	classifier.print_meta()
if __name__ == '__main__':
	np.set_printoptions(threshold=sys.maxsize)
	main()
