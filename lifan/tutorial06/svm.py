import sys
from collections import defaultdict

def create_features(x):
	phi = defaultdict(int)
	words = x.split(" ")
	for word in words:
			phi["UNI:"+word] += 1
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

def update_weights(w,phi,y, c):
	def sign(x):
		if x < 0:
			return -x
		elif x > 0:
			return x
		return 0

	for name, value in w.items():
		if abs(value) <= c:
			w[name] = 0
		else:
			w[name] -= sign(value) * c
	for name, value in phi.items():
		w[name] += value * y

def train_svm(input_file, margin):
	w = defaultdict(float)
	with open(input_file, 'r') as fin:
		for line in fin:
			splited_line = line.strip().split('\t')
			x = splited_line[1]
			y = float(splited_line[0])
			phi = create_features(x)
			score = predict_one(w, phi)
			val = score * y
			if val <= margin:
				update_weights(w, phi, y, 0.0001)
	with open("model.txt", 'w') as fout:
		for key, value in w.items():
			fout.write(f"{key}\t{value}\n")

def test_svm(test_file, model_file, output_file):
	w = defaultdict(float)
	with open(model_file, 'r') as fmodel, open(test_file, 'r') as ftest, open(output_file, 'w') as fout:
		for line in fmodel:
			splited_line = line.strip().split("\t")
			w[splited_line[0]] = float(splited_line[1])

		for line in ftest:
			sentence = line.strip()
			phi = create_features(sentence)
			y_2 = predict_one(w, phi)
			fout.write(f"{y_2}\t{sentence}\n")

if __name__ == '__main__':
	input_file = '../../data/titles-en-train.labeled'
	model_file = 'model.txt'
	test_file = '../../data/titles-en-test.word'
	output_file = 'test_result.txt'
	train_svm(input_file, 20)
	test_svm(test_file, model_file, output_file)

# ../../script/grade-prediction.py ../../data/titles-en-test.labeled test_result.txt
# Accuracy = 80.977683%
