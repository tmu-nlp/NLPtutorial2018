import numpy as np
from collections import defaultdict
import pickle
from tqdm import tqdm

def create_features(sentence, ids):
	phi = np.zeros(len(ids))
	words = sentence.split()
	for word in words:
			phi[ids["UNI:"+word]] += 1
	return phi

def init_network(feature_size, layer, node):
	w0 = np.random.rand(node, feature_size) / 5 - 0.1
	b0 = np.random.rand(1, node) / 5 - 0.1
	net = [(w0, b0)]

	while len(net) < layer:
		w = np.random.rand(node, node) / 5 - 0.1
		b = np.random.rand(1, node) / 5 - 0.1
		net.append((w, b))

	w_o = np.random.rand(1, node) / 5 - 0.1
	b_o = np.random.rand(1, 1) / 5 - 0.1
	net.append((w_o, b_o))

	return net

def forward_nn(net, phi0):
	phi = [0 for _ in range(len(net) + 1)]
	phi[0] = phi0
	for i in range(len(net)):
		w, b = net[i]
		phi[i + 1] = np.tanh(np.dot(w, phi[i]) + b).T
	return phi

def backward_nn(net, phi, label):
	J = len(net)
	delta = np.zeros(J + 1, dtype=np.ndarray)
	delta[-1] = np.array([label - phi[J][0]])
	delta_p = np.zeros(J + 1, dtype=np.ndarray)

	for i in range(J, 0, -1):
		delta_p[i] = (1 - np.square(phi[i])).T * delta[i]
		w, _ = net[i - 1]
		delta[i - 1] = np.dot(delta_p[i], w)

	return delta_p

def update_weights(net, phi, delta, lambda_):
	for i in range(len(net)):
		w, b = net[i]
		w += lambda_ * np.outer(delta[i + 1], phi[i])
		b += lambda_ * delta[i + 1]

def train_nn(train_file, output_file, lambda_=0.1, epoch=1, hidden_l=1, hidden_n=2):
	ids = defaultdict(lambda: len(ids))
	feature_labels = []

	for line in open(train_file, encoding='utf8'):
		_, sentence = line.strip().split('\t')
		for word in sentence.split():
			ids['UNI:' + word]

	for line in open(train_file, encoding='utf8'):
		label, sentence = line.strip().split('\t')
		label = int(label)
		phi = create_features(sentence, ids)
		feature_labels.append((phi, label))

	net = init_network(len(ids), hidden_l, hidden_n)

	for _ in tqdm(range(epoch)):
		for phi0, label in feature_labels:
			phi = forward_nn(net, phi0)
			delta = backward_nn(net, phi, label)
			update_weights(net, phi, delta, lambda_)

	with open(output_file, 'wb') as f:
		pickle.dump(net, f)
		pickle.dump(dict(ids), f)

if __name__ == '__main__':
	train_file = '../../data/titles-en-train.labeled'
	model_file = 'model.txt'

	train_nn(train_file, model_file)