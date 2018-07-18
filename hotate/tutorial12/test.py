# -*- coding: utf-8 -*-
from train_hmm_percep import viterbi
import pickle


def main(path):
    w = pickle.load(open('weight_10', 'rb'))
    transition = pickle.load(open('transition', 'rb'))
    tags = pickle.load(open('tags', 'rb'))
    x_data = load_data(path)

    with open('result', 'w') as f:
        for x in x_data:
            y = viterbi(w, x, transition, tags)
            print(' '.join(y), file=f)


def load_data(path):
    x_data = []
    for line in open(path, 'r'):
        x_data.append(line.strip().split())
    return x_data


if __name__ == '__main__':
    # main('../../test/05-test-input.txt')
    main('../../data/wiki-en-test.norm')

# Accuracy: 88.67% (4046/4563)
#
# Most common mistakes:
# NNS --> NN      55
# NN --> JJ       36
# JJ --> NN       34
# VBN --> NN      23
# NNP --> NN      22
# NNP --> JJ      17
# RB --> JJ       13
# VBN --> JJ      12
# NN --> NNP      11
# NN --> VBG      11


