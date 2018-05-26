# -*- coding: utf-8 -*-
import os, sys
from pathlib import Path
import csv

tutorial01_path = str(Path(os.getcwd()).parent.joinpath('tutorial01'))
sys.path.append(tutorial01_path)

from TrainUnigram import *
from TestUnigram import *
from DevideWords import DevideWords


def test(_file_name):
    obj = DevideWords()
    obj.import_model(_file_name, deliminator='\t')
    with open('../../test/04-input.txt', encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n', '')
            best_scores, best_edges = obj.viterbi_forward(line)
            words = obj.viterbi_backward(line, best_edges)
            print(' '.join(words))


def train(_name):
    trainer = TrainUnigram()
    tester = TestUnigram()
    input_file = '../../data/' + _name + '-train.word'
    model_file = _name + '-model.txt'
    trainer.train_model(input_file, model_file)
    tester.import_model(model_file)
    tester.evaluate(input_file)


def evaluate(_input_file, _model_file):
    ret = []
    obj = DevideWords()
    _model_file = _model_file + '-model.txt'
    obj.import_model(_model_file, deliminator=',')
    with open('../../data/' + _input_file + '.txt', encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n', '')
            best_scores, best_edges = obj.viterbi_forward(line)
            words = obj.viterbi_backward(line, best_edges)
            if len(words) != 0:
                print(' '.join(words))
                ret.append(' '.join(words))
    return ret


## train
print("---- test with example ----")
test('../../test/04-model.txt')
# print("---- train by samples ----")
# train('wiki-ja-train')
# print("---- apply for the sample ----")
# ret = evaluate('wiki-ja-test', 'wiki-ja')
# with open('my_answer.word', 'w', encoding='utf-8') as f:
#     f.write('\n'.join(ret))
