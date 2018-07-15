# -*- coding: utf-8 -*-
from collections import defaultdict
import pickle
from tqdm import tqdm


def main(path):
    x_data, y_prime = make_train_data(path)
    transition, possible_tags = load_model(x_data, y_prime)
    w = defaultdict(int)
    epoch = 5
    for e in tqdm(range(epoch)):
        for x, y in zip(x_data, y_prime):
            y_hat = viterbi(w, x, transition, possible_tags)
            phi_prime = create_feature(x, y)
            phi_hat = create_feature(x, y_hat)
            update_weight(w, phi_prime, 1)
            update_weight(w, phi_hat, -1)

    pickle.dump(w, open('weight_5', 'wb'))
    pickle.dump(transition, open('transition', 'wb'))
    pickle.dump(possible_tags, open('tags', 'wb'))


def update_weight(w, dic, cal):
    for key, value in dic.items():
        w[key] += cal * value


def create_feature(x, y):
    phi = defaultdict(int)
    for i in range(len(y) + 1):
        if i == 0:
            first_tag = '<s>'
        else:
            first_tag = y[i - 1]

        if i == len(y):
            next_tag = '</s>'
        else:
            next_tag = y[i]

        phi[create_trans(first_tag, next_tag)] += 1

    for i in range(len(y)):
        phi[create_emit(y[i], x[i])] += 1

    return phi


def create_emit(tag, word):
    emit = f'E|{tag}|{word}'
    return emit


def create_trans(tag1, tag2):
    trans = f'T|{tag1}|{tag2}'
    return trans


def make_train_data(path):
    word_data = []
    pos_data = []
    for line in open(path, 'r'):
        word_pos = [w.split('_') for w in line.strip().split()]
        word_data.append([w[0] for w in word_pos])
        pos_data.append([p[1] for p in word_pos])
    return word_data, pos_data


def load_model(word_data, tag_data):
    transition = defaultdict(lambda: 0)
    possible_tags = set()
    possible_tags.add('<s>')
    possible_tags.add('</s>')
    for word, pos in zip(word_data, tag_data):

        transition[f'<s>|{pos[0]}'] += 1
        for p in range(len(pos)-1):
            transition[f'{pos[p]}|{pos[p+1]}'] += 1
        transition[f'{pos[-1]}|</s>'] += 1

        for p in pos:
            possible_tags.add(p)

    return dict(transition), possible_tags


def viterbi(w, words, transition, possible_tags):
    words.append('</s>')
    length = len(words)
    best_edge = {f'0|<s>': None}
    best_score = {f'0|<s>': 0}
    for i in range(length):
        for prev_tag in possible_tags:
            if prev_tag == '</s>':
                continue
            for next_tag in possible_tags:
                i_prev = f'{i}|{prev_tag}'
                prev_next = f'{prev_tag}|{next_tag}'
                i_1_next = f'{i+1}|{next_tag}'
                if i_prev in best_score.keys() and prev_next in transition:
                    score = best_score[i_prev]
                    score += w[create_trans(prev_tag, next_tag)]
                    score += w[create_emit(next_tag, words[i])]
                    if i_1_next not in best_score.keys() or best_score[i_1_next] < score:
                        best_score[i_1_next] = score
                        best_edge[i_1_next] = i_prev

    tags = []
    next_edge = best_edge[f'{len(words)}|</s>']
    while next_edge != '0|<s>':
        tag = next_edge.split('|')[1]
        tags.append(tag)
        next_edge = best_edge[next_edge]
    tags.reverse()
    return tags


if __name__ == '__main__':
    # main('../../test/05-train-input.txt')
    main('../../data/wiki-en-train.norm_pos')
