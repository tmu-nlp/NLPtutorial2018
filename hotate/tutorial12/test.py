# -*- coding: utf-8 -*-
from train_hmm_percep import viterbi
import pickle


def main(path):
    w = pickle.load(open('weight', 'rb'))
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
