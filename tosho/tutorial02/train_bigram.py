import os, sys
sys.path.append(os.path.pardir)
from common.utils import count_words
from common.n_gram import ZeroGram, NGram

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_train_file')
    parser.add_argument('path_to_cache_file')
    parser.add_argument('-n', '--n-gram', type=int, required=True)
    arg = parser.parse_args()

    model = NGram(arg.n_gram)
    model.train(arg.path_to_train_file)

'''
p(b|a) = l * p(b|a) + (1 - l) * p(b)
p(b) = m * p(b) + (1 - m) * p(unk)

words['2gram']
{
    p(b|a) : 0.8
    p(c|a) : 0.2
}
words['1gram']
{
    p(a) : 0.7
    p(b) : 0.2
    p(c) : 0.1
}
words['0gram']
{
    p(unk) = 1.0
}


train model with ../../data/wiki-en-train.word
5234 words
p(unk) = 5.0000000000000004e-08
p(,) = 0.04720584208749511
p(the) = 0.03830008906032029
p(</s>) = 0.03448333776295966
p(.) = 0.03397973863344679
p(of) = 0.029738903858601638
p(a) = 0.021601802134367503
p(to) = 0.019958478659115007
p(and) = 0.01834166040120529
p(in) = 0.0141538360610457
p(is) = 0.013040616932648847
saved parameters to wiki-en-train.pyc
'''