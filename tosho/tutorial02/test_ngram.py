import os, sys
sys.path.append(os.path.pardir)
from common.n_gram import NGram

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--path_to_cache_file')
    parser.add_argument('-t', '--path_to_test_file')
    arg = parser.parse_args()

    model = NGram()
    model.load(arg.path_to_cache_file)
    model.print_params()

    print(f'loaded model({model.n}-gram) from {arg.path_to_cache_file}')
    print(f'test model with {arg.path_to_test_file}')

    entropy = model.entropy(arg.path_to_test_file)
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


'''