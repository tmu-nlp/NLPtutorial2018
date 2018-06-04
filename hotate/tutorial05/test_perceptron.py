# -*- coding: utf-8 -*-

from collections import defaultdict
from train_perceptron import create_features, predict_one


def load_model(model_file):
    w = defaultdict(lambda: 0)
    for line in model_file:
        line = line.split('\t')
        n_gram = line[0]
        weight = line[1].strip('\n')
        w[n_gram] = int(weight)
    return w


def predict_all(model_file, input_fle, n):
    w = load_model(model_file)
    result = []
    for sentence in input_fle:
        phi = defaultdict(lambda: 0)
        phi = create_features(sentence.lower(), n, phi)
        result.append(predict_one(w, phi))
    return result


if __name__ == '__main__':
    input_file = open('../../data/titles-en-test.word', 'r')
    mode_file = open('model_file', 'r')
    with open('my_answer', 'w') as f:
        result = predict_all(mode_file, input_file, 2)
        for line in result:
            print(line, file=f)

# python grade-prediction.py data/titles-en-test.labeled ../hotate/tutorial05/my_answer
# Accuracy = 93.942614% l = 20, n = 1

# Accuracy = 94.190577% l = 30, n = 2
# Accuracy = 94.261424% l = 40, n = 2
# Accuracy = 94.544810% l = 40, n = 2 (remove_symbol)
# Accuracy = 94.261424% l = 50, n = 2

# Accuracy = 94.048884% l = 20, n = 3
# Accuracy = 94.261424% l = 30, n = 3
