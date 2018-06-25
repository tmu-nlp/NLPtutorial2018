import numpy as np

#sigmoid関数
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

#２乗和誤差
def mean_squared_error(y, t):
	return 0.5 * np.sum((y-t)**2)

#交差エントロピー誤差
def cross_entropy_error(y, t):
	delta = 1e-7
	return -np.sum(t * np.log(y + delta))

def softmax(a):
	c = np.max(a)
	exp_a = np.exp(a - c)
	y = exp_a / np.sum(exp_a)
	return y

