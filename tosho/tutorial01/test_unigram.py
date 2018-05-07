import os, sys
sys.path.append(os.path.pardir)
from tutorial01.train_unigram import UniGram

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_cache_file')
    parser.add_argument('path_to_test_file')
    arg = parser.parse_args()

    model = UniGram()
    model.load(arg.path_to_cache_file)

    print(f'test model with {arg.path_to_test_file}')

    entropy, coverage, perplexity = model.estimate(arg.path_to_test_file)

    print(f'entropy = {entropy}')
    print(f'converage = {coverage}')
    print(f'perplexity = {perplexity}')

    
'''
test model with ../../data/wiki-en-test.word
entropy = 10.527337238682652
converage = 0.895226024503591
perplexity = 1475.8570129853622
'''