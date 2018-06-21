# -*- coding: utf-8 -*-

from collections import defaultdict
from train import create_features, predict_one


def load_model(model_file):
    w = defaultdict(lambda: 0)
    for line in model_file:
        line = line.split('\t')
        n_gram = line[0]
        weight = line[1].strip('\n')
        w[n_gram] = float(weight)
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

# python grade-prediction.py data/titles-en-test.labeled ../hotate/tutorial06/my_answer


# Accuracy = 94.544810% epoch = 10, n = 2
