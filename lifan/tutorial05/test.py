import sys
from collections import defaultdict
from train_perceptron import create_features, predict_one

with open(sys.argv[1], "r") as model_file, open(sys.argv[2], "r") as input_file, open("test_result.txt", "w") as fout:
	w = defaultdict(int)
	for line in model_file:
		splited_line = line.strip().split("\t")
		w[splited_line[0]] = int(splited_line[1])
	for x in input_file:
		phi = create_features(x.strip())
		y_2 = predict_one(w, phi)
		fout.write(f"{y_2}\n")

# Accuracy = 80.517180%