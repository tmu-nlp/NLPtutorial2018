import os, sys
sys.path.append(os.path.pardir)
from common.n_gram import NGram
from common.smoothings import SimpleSmoothing, MultiLayerSmoothing, WittenBell

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test-file')
    parser.add_argument('-p', '--param-file')
    arg = parser.parse_args()

    model = NGram()
    model.load_params(arg.param_file)
    model.print_params()

    smoothing = SimpleSmoothing()
    # blender = MultiLayerSmoothing(unk_rates={
    #     1: 0.1, 
    #     2: 0.1
    # })
    # blender = WittenBell(arg.path_to_train_file)
    model.set_smoothing(smoothing)

    print(f'loaded model({model.n}-gram) from {arg.param_file}')
    print(f'test model with {arg.test_file}')

    with open(arg.test_file, 'r') as f:
        entropy = model.entropy(f)

    print(f'entropy = {entropy}')

'''
$ python test_ngram.py -c test-bigram.pyc -t ../../test/02-train-input.txt
6 types (2-gram)
p(('<s>', 'a')) = 0.25
p(('a', 'b')) = 0.25
p(('b', 'c')) = 0.125
p(('c', '</s>')) = 0.125
p(('b', 'd')) = 0.125
p(('d', '</s>')) = 0.125
6 types (1-gram)
p(('<s>',)) = 0.2
p(('a',)) = 0.2
p(('b',)) = 0.2
p(('</s>',)) = 0.2
p(('c',)) = 0.1
p(('d',)) = 0.1
p(unk) = 1e-06
loaded model(2-gram) from test-bigram.pyc
test model with ../../test/02-train-input.txt
entropy = 2.5038050854389615

$ python test_ngram.py -c wiki-bigram.pyc -t ../../data/wiki-en-test.word
21514 types (2-gram)
p(('.', '</s>')) = 0.03532168963785503
p(('of', 'the')) = 0.005440544612465822
p((',', 'and')) = 0.005273143239774566
p(('<s>', 'The')) = 0.004073433402153897
p(('in', 'the')) = 0.0038781318006807656
p((',', 'the')) = 0.0029016237933151053
p(('-RRB-', '.')) = 0.002817923106969477
p(('<s>', 'In')) = 0.002706322191841973
p(('of', 'a')) = 0.002566821047932593
p(('can', 'be')) = 0.002538920819150717
5235 types (1-gram)
p((',',)) = 0.04794981557763239
p(('the',)) = 0.038903696524244136
p(('<s>',)) = 0.03502678835850631
p(('</s>',)) = 0.03502678835850631
p(('.',)) = 0.034515251864415904
p(('of',)) = 0.030207576124707213
p(('a',)) = 0.021942223299141157
p(('to',)) = 0.020272998950004038
p(('and',)) = 0.018630697574240098
p(('in',)) = 0.014376867781277764
p(unk) = 1e-06
loaded model(2-gram) from wiki-bigram.pyc
test model with ../../data/wiki-en-test.word
entropy = 14.111808618953058
'''