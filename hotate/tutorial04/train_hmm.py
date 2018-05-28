# -*- coding: utf-8 -*-

from collections import defaultdict


def train_hmm(path):
    emit = defaultdict(lambda: 0)
    transition = defaultdict(lambda: 0)
    context = defaultdict(lambda: 0)
    for line in open(path, 'r'):
        line = line.strip('\n')
        previous = '<s>'
        context[previous] += 1
        word_tags = line.split(' ')
        for word_tag in word_tags:
            word_tag = word_tag.split('_')
            word = word_tag[0].lower()
            tag = word_tag[1]
            # 遷移の数え上げ
            transition[previous + ' ' + tag] += 1
            # 出現頻度の数え上げ
            context[tag] += 1
            # 生成の数え上げ
            emit[tag + ' ' + word] += 1
            previous = tag
        # 文末の遷移
        transition[previous + ' </s>'] += 1

    with open('model_file', 'w') as f:
        # 遷移確率を出力
        for key, transition_count in transition.items():
            previous = key.split(' ')[0]
            f.write(f'T {key} {transition_count / context[previous]}\n')
        for key, emit_count in emit.items():
            previous = key.split(' ')[0]
            f.write(f'E {key} {emit_count / context[previous]}\n')


if __name__ == '__main__':
    # path = '../../test/05-train-input.txt'
    path = '../../data/wiki-en-train.norm_pos'
    train_hmm(path)
