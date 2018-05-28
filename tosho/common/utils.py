import sys
from collections import defaultdict
import random

def load_labeled_data(file_name):
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip('\n')
            t, x = line.split('\t')
            x = [word.lower() for word in x.split(' ')]
            yield (x, int(t))

def load_word_data(file_name):
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip('\n')
            x = [word.lower() for word in line.split(' ')]
            yield x

def count_tokens(tokens):
    counts = defaultdict(int)
    for token in tokens:
        counts[token] += 1
    return counts

def count_words(target_filename, n_gram=1):
    with open(target_filename, 'r') as target_file:
        if n_gram == 1:
            return count_unigram_words(target_file)
        else:
            return count_n_gram_words(target_file, n_gram)

def count_n_gram_words(target, n):
    word_count = defaultdict(int)
    for seq in parse_file(target, n):
        for pair in seq:
            word_count[pair] += 1
    return word_count

def count_unigram_words(target):
    # 配列で渡した文字列も要素ごとに比較してくれる
    word_count = defaultdict(int)
    for tokens_in_line in parse_file(target, 1):
        for token in tokens_in_line:
            word_count[token] += 1
    return word_count

def parse_file(src, n_gram):
    for tokens_in_line in parse_unigram_file(src):
        # １つずつズラした配列を作成する
        yield zip(*[tokens_in_line[i:] for i in range(n_gram)])

def parse_unigram_file(src):
    for line in src:
        line = line.strip()
        if len(line) == 0:
            continue
        else:
            words = line.split()
            words.append('</s>')
            words.insert(0, '<s>')
            yield words

def load_data(file_name, n):
    return list(iterate_data(file_name, n))

def iterate_data(file_name, n):
    with open(file_name, 'r') as f:
        for token in iterate_tokens(f, n):
            yield token

def iterate_tokens(doc, n):
    '''
    doc : list or iterator
    ドキュメント内に含まれるn-gramトークンを列挙する
    '''
    for line in iterate_lines(doc):
        words = list(iterate_words(line))
        for token in zip(*[words[i:] for i in range(n)]):\
            yield token

def iterate_lines(doc):
    for line in doc:
        line = line.strip()
        if len(line) == 0:
            continue
        else:
            yield line

def iterate_words(line):
    yield '<s>'
    for word in line.split():
        yield word
    yield '</s>'

if __name__ == '__main__':
    tokens = load_data('../../test/02-train-input.txt', 2)
    tokens2 = iterate_data('../../test/02-train-input.txt', 2)
    for pair in zip(tokens, tokens2):
        print(f'{pair} | {pair[0] == pair[1]}')
    