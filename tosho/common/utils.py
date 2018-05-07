import sys
from collections import defaultdict
import random

def count_words(target_filename, n_gram=1):
    ret = defaultdict(int)

    with open(target_filename, 'r') as target_file:
        if n_gram == 1:
            return count_unigram_words(target_file)
        else:
            return count_n_gram_words(target_file, n_gram)

def count_n_gram_words(target, n):
    word_count = defaultdict(int)
    for tokens_in_line in parse_file(target):
        # １つずつズラした配列を作成する
        seq = zip(*[tokens_in_line[i:] for i in range(n)])
        for pair in seq:
            word_count[pair] += 1
    return word_count

def count_unigram_words(target):
    # 配列で渡した文字列も要素ごとに比較してくれる
    word_count = defaultdict(int)
    for tokens_in_line in parse_file(target):
        for token in tokens_in_line:
            word_count[token] += 1
    return word_count

def parse_file(src):
    for line in src:
        line = line.strip()
        if len(line) == 0:
            continue
        else:
            words = line.split()
            words.append('</s>')
            words.insert(0, '<s>')
            yield words