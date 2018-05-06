import os, sys
sys.path.append(os.path.pardir)
from tutorial00.word_count import load_file
from tutorial01.train_unigram import UniGram

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_cache_file')
    parser.add_argument('path_to_test_file')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--includes-eos', action='store_true', default=False)
    arg = parser.parse_args()

    model = UniGram(arg.debug)
    model.load(arg.path_to_cache_file)

    test_file = arg.path_to_test_file

    print('testing model...')

    test_data = open(test_file, 'r')
    lines = load_file(test_data, arg.includes_eos)

    model.estimate(lines)

    print(f'entropy = {model.entropy}')
    print(f'converage = {model.coverage}')
    print(f'perplexity = {model.perplexity}')

    test_data.close()    
    