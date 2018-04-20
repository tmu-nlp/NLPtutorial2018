# -*- coding: utf-8 -*-
from TrainBigram import *
from TestBigram import *

## train
print("---- train with sample ----")
trainer = TrainBigram()
tester = TestBigram()
input_file = '../../test/02-train-input.txt'
model_file = 'train-input-model.txt'
trainer.train_model(input_file, model_file)
tester.import_model(model_file)
tester.evaluate(input_file)

## evaluate
print('---- train with wiki data----')
trainer = TrainBigram()
tester = TestBigram()
input_file = '../../data/wiki-en-train.word'
model_file = 'wiki-en-model.txt'
trainer.train_model(input_file, model_file)
tester.import_model(model_file)
tester.evaluate(input_file)