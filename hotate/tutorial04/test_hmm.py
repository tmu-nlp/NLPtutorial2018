# -*- coding: utf-8 -*-

import math
from collections import defaultdict


def load_model(model_file, path):
    transition = defaultdict(lambda: 0)
    emission = defaultdict(lambda: 0)
    possible_tags = defaultdict(lambda: 0)
    for line in model_file:
        line = line.strip('\n').split(' ')
        type = line[0]
        context = line[1]
        word = line[2]
        prob = line[3]
        possible_tags[context] = 1
        if type == 'T':
            transition[f'{context} {word}'] = float(prob)
        else:
            emission[f'{word} {context}'] = float(prob)

    for line in open(path, 'r'):
        lam = 0.95
        V = 1000000
        words = line.lower().strip('\n').split()
        words.append('</s>')
        l = len(words)
        best_score = defaultdict(lambda: 10 ** 10)
        best_edge = defaultdict(str)
        best_score['0 <s>'] = 0
        best_edge['0 <s>'] = None
        for i in range(0, l):
            for prev_tag in possible_tags.keys():
                for next_tag in possible_tags.keys():
                    if best_score[f'{i} {prev_tag}'] is not 10 ** 10 and transition[f'{prev_tag} {next_tag}'] is not 0:
                        score = best_score[f'{i} {prev_tag}'] \
                                + -math.log(transition[f'{prev_tag} {next_tag}']) \
                                + -math.log((lam * emission[f'{words[i]} {next_tag}']) + ((1-lam) / V))
                        if best_score[f'{i+1} {next_tag}'] > score:
                            best_score[f'{i+1} {next_tag}'] = score
                            best_edge[f'{i+1} {next_tag}'] = f'{i} {prev_tag}'

                next_tag = '</s>'
                if best_score[f'{i} {prev_tag}'] is not 10 ** 10 and transition[f'{prev_tag} {next_tag}'] is not 0:
                    score = best_score[f'{i} {prev_tag}'] + -math.log(transition[f'{prev_tag} {next_tag}'])
                    if best_score[f'{i+1} {next_tag}'] > score:
                        best_score[f'{i+1} {next_tag}'] = score
                        best_edge[f'{i+1} {next_tag}'] = f'{i} {prev_tag}'

        tags = []
        next_edge = best_edge[f'{len(words)} </s>']
        while next_edge is not '0 <s>':
            if next_edge == '0 <s>':
                break
            tag = next_edge.split(' ')[1]
            tags.append(tag)
            next_edge = best_edge[next_edge]
        tags.reverse()
        sentence = ' '.join(tags)
        yield sentence


if __name__ == '__main__':
    # path = '../../test/05-test-input.txt'
    path = '../../data/wiki-en-test.norm'
    model_file = open('model_file')
    with open('my_answer.word', 'w') as f:
        for sentence in load_model(model_file, path):
            f.write(f'{sentence}\n')

'''
Accuracy: 90.86% (4146/4563)

Most common mistakes:
NNS --> NN      44
NNP --> NN      29
NN --> JJ       27
JJ --> DT       18
NNP --> JJ      15
JJ --> NN       12
VBN --> NN      11
VBN --> JJ      10
NN --> IN       9
NN --> DT       7
'''



