import numpy as np
from common import *

class TwoLayerNet:

	def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
		self.network = {}
		self.network['W1'] = np.random.randn(input_size, hidden_size) / 5 - 0.1
		self.network['b1'] = np.zeros(1, hidden_size) / 5 - 0.1
		self.network['W2'] = np.random.randn(hidden_size, output_size) / 5 - 0.1
		self.network['b2'] = np.zeros(1, output_size) / 5 - 0.1

	def predict(self, x):
		W1, W2 = self.network['W1'], self.network['W2']
		b1, b2 = self.network['b1'], self.network['b2']

		a1 = np.dot(x, W1) + b1
		z1 = sigmoid(a1)
		a2 = np.dot(z1, W2) + b2
		y = softmax(a2)

		return y

	def forward(self, x):
		W1, W2 = self.network['W1'], self.network['W2']
		b1, b2 = self.network['b1'], self.network['b2']
		a1 = np.dot(x, W1) + b1
		z1 = np.tanh(a1)
		a2 = np.dot(z1, W2) + b2
		out = np.tanh(a2)

		return out

	def backward(self, dout, label):
		delta = np.zeros(2)
		delta[-1] = np.array()

	# x:input_data, t:label_data
	def loss(self, x, t):
		y = self.predict(x)
		 return cross_entropy_error(y, t)

