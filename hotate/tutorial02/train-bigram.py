# -*- coding: utf-8 -*-

from collections import defaultdict


def bigram_model(path):
    count_ngram = defaultdict(lambda: 0)  # {ngram} : {回数}
    counts = defaultdict(lambda: 0)  # {ngramの先頭の単語} : {回数}
    u = defaultdict(lambda: 0)  # 単語の異なり数
    V = 0  # 全体の単語数
    T = 0  # unigramの異なり数
    for line in open(path, 'r'):
        words = line.strip().split(" ")
        words.append('</s>')
        words.insert(0, '<s>')
        for i in range(1, len(words)):
            # bigram
            count_ngram[words[i - 1] + ' ' + words[i]] += 1
            if count_ngram[words[i - 1] + ' ' + words[i]] == 1:  # 平滑化で使う単語の異なり数
                u[words[i - 1]] += 1
            counts[words[i - 1]] += 1
            # unigram
            count_ngram[words[i]] += 1
            if count_ngram[words[i]] == 1:
                T += 1
            V += 1

    lam_1 = T / (T + V)  # T:単語の異なり数
    lam_2 = witten_bell(count_ngram, counts, u)

    with open('model_file', 'w') as model_file:
        for ngram, count in sorted(count_ngram.items()):
            word = ngram.split(' ')
            if len(word) == 1:
                probability = float(count / V)
                model_file.write(f'{ngram}\t{probability}\t{lam_1}\n')
            else:
                probability = float(count / counts[word[0]])
                model_file.write(f'{ngram}\t{probability}\t{lam_2[word[0]]}\n')


def witten_bell(count_ngram, counts, u):
    lam = defaultdict(lambda: 0)
    for ngram in count_ngram:
        words = ngram.split(' ')
        if len(words) > 1:
            lam[words[0]] = counts[words[0]] / (u[words[0]] + counts[words[0]])

    return lam


if __name__ == '__main__':
    path = '../../data/wiki-en-train.word'
    #  path = '../../test/02-train-input.txt'
    bigram_model(path)
