# -*- coding: utf-8 -*-
import pickle
from RNN import Output
from train_rnn import create_one_hot, forward


def test(path):
    x_ids = pickle.load(open('x_ids.pkl', 'rb'))
    y_ids = pickle.load(open('y_ids.pkl', 'rb'))
    x_size = len(x_ids)
    feat_lab = []
    for sentence in open(path, 'r'):
        x_list = []
        for x in sentence.lower().split():
            x_vec = create_one_hot(x, x_size, x_ids)
            x_list.append(x_vec)
        feat_lab.append(x_list)

    out = Output()
    hid = pickle.load(open('hidden_model_84_30_0.005', 'rb'))
    inp = pickle.load(open('input_model_84_30_0.005', 'rb'))

    with open('my_answer', 'w') as f:
        for x_list in feat_lab:
            forward(x_list, out, hid, inp)
            pred_list = out.answer(y_ids)
            pred_str = ' '.join(pred_list)
            print(pred_str, file=f)


if __name__ == '__main__':
    # test('../../test/05-test-input.txt')
    test('../../data/wiki-en-test.norm')

# perl gradepos.pl data/wiki-en-test.pos ../hotate/tutorial08/my_answer
#
# hidden_size = 84
# epoch = 30
# lamda = 0.005
# Accuracy: 89.52% (4088/4563)
#
# Most common mistakes:
# NNS --> NN      65
# JJ --> NN       64
# NNP --> NN      51
# VBN --> NN      23
# IN --> WDT      17
# VB --> NN       15
# VBP --> NN      15
# VBG --> NN      15
# NN --> JJ       13
# JJ --> NNS      11
