import numpy as np
from collections import defaultdict
from two_layer_net import TwoLayerNet

def 

def create_features(sentence, ids):
	phi = np.zeros(len(ids))
	words = sentence.split()
	for word in words:
			phi[ids["UNI:"+word]] += 1
	return phi


def predict_one(w,phi):
	score = 0
	for name,value in phi.items():
		if name in w:
			score += value * w[name]
	if score >= 0:
		return 1
	else:
		return -1

def train(train_file, hidden_size=1, output_size=2):
	ids = defaultdict(lambda: len(ids))
	feature_labels = []
	for line in open(train_file, encoding='utf8'):
		sentence = line.strip().split('\t')[1]
		for word in sentence.split():
			ids['UNI:' + word]

	for line in open(train_file, encoding='utf8'):
		label, sentence = line.strip().split('\t')
		label = int(label)
		phi = create_features(sentence, ids)
		feature_labels.append(phi, label)

	network = TwoLayerNet(len(ids), hidden_size, output_size)

	
	

if __name__ == '__main__':
	train_file = '../../data/titles-en-train.labeled'
	model_file = 'model.txt'
	
		print(ids)

	# with open(train_file, 'r') as f:
	# 	for sentence in f:
	# 		print(create_features(sentence, ids))