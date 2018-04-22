# -*- coding: utf-8 -*-
from TrainUnigram import *
from TestUnigram import *

## train
print("---- train with sample ----")
trainer = TrainUnigram()
tester = TestUnigram()
input_file = '../../test/01-train-input.txt'
model_file = 'train-input-model.txt'
trainer.train_model(input_file, model_file)
tester.import_model(model_file)

## test
print("---- test by sample----")
trainer = TrainUnigram()
tester = TestUnigram()
input_file = '../../test/01-test-input.txt'
model_file = 'train-input-model.txt'
tester.import_model(model_file)
tester.evaluate(input_file)

## evaluate
print('---- train with wiki data----')
trainer = TrainUnigram()
tester = TestUnigram()
input_file = '../../data/wiki-en-train.word'
model_file = 'wiki-en-model.txt'
trainer.train_model(input_file, model_file)
tester.import_model(model_file)
tester.evaluate(input_file)

print('---- test by wiki data----')
trainer = TrainUnigram()
tester = TestUnigram()
input_file = '../../data/wiki-en-test.word'
model_file = 'wiki-en-model.txt'
tester.import_model(model_file)
tester.evaluate(input_file)