import argparse
import pickle
from collections import defaultdict
from itertools import product
from tqdm import tqdm as tq


def arguments_parse():
    parser = argparse.ArgumentParser(
        usage='\t品詞推定のための学習プログラム',
        description='説明',
        epilog='end',
        add_help=True,
    )
    parser.add_argument('-t', '--train', help='学習用ファイル名', type=str)
    parser.add_argument('-o', '--output', help='出力ファイル名', type=str)
    parser.add_argument('-i', '--input', help='入力ファイル名', type=str)
    return parser.parse_args()


def create_trans(tag1, tag2):
    '遷移に対する素性のキーのリストを返す'
    return [f'T,{tag1},{tag2}']


def create_emit(tag, word):
    '生成に対する素性のキーのリストを返す'
    keys = [f'E,{tag},{word}']
    if word[0].isupper():
        keys.append(f'CAPS,{tag}')
    return keys


def create_features(X, Y):
    '素性を作成して返す'
    phi = defaultdict(int)
    for i in range(len(Y) + 1):
        first_tag = Y[i - 1] if i else '<s>'
        next_tag = Y[i] if i != len(Y) else '</s>'
        for key in create_trans(first_tag, next_tag):
            phi[key] += 1
        if i == len(Y):
            break
        for key in create_emit(Y[i], X[i]):
            phi[key] += 1
    return phi


def load_train_data(epoch=1):
    for _ in tq(range(epoch)):
        for line in open(train_file, encoding='utf8'):
            wordtags = line.split()
            X = []
            Y = []
            for wordtag in wordtags:
                word, tag = wordtag.split('_')
                X.append(word)
                Y.append(tag)
            yield X, Y


def load_test_data():
    for line in open(test_file, encoding='utf8'):
        X = line.strip().split()
        yield X


def update_best(score, prev, next_, best_score, best_edge):
    '最良のスコアとエッジを更新'
    if next_ not in best_score or best_score[next_] < score:
        best_score[next_] = score
        best_edge[next_] = prev


def hmm_viterbi(w, X, tags, transition):
    # 前向きステップ
    l = len(X)
    # BOS
    best_score = {'0 <s>': 0}
    best_edge = {'0 <s>': None}
    # Sequense
    for i, prev, next_ in product(range(l), tags, tags):
        if f'{i} {prev}' not in best_score or f'{prev} {next_}' not in transition:
            continue
        score = best_score[f'{i} {prev}'] + sum(w[key] for key in create_trans(prev, next_) + create_emit(next_, X[i]))
        update_best(score, f'{i} {prev}', f'{i+1} {next_}', best_score, best_edge)
    # EOS
    for tag in tags:
        if not transition[f'{tag} </s>']:
            continue
        score = best_score[f'{l} {tag}'] + sum(w[key] for key in create_trans(tag, '</s>'))
        update_best(score, f'{l} {tag}', f'{l+1} </s>', best_score, best_edge)

    # 後ろ向きステップ
    tag_path = []
    next_edge = best_edge[f'{l+1} </s>']
    while next_edge != '0 <s>':
        _, tag = next_edge.split()
        tag_path.append(tag)
        next_edge = best_edge[next_edge]
    tag_path.reverse()

    return tag_path


def update(w, phi_prime, phi_hat):
    for key, value in phi_prime.items():
        w[key] += value
    for key, value in phi_hat.items():
        w[key] -= value


def train_hmm_percep():
    # 学習データ内の全遷移と全POSを取得
    transition = defaultdict(int)
    possible_tags = {'<s>', '</s>'}
    for _, tags in load_train_data():
        for p, n in zip(['<s>'] + tags, tags + ['</s>']):
            transition[f'{p} {n}'] += 1
        possible_tags.update(set(tags))

    w = defaultdict(int)
    for X, Y_prime in load_train_data(5):
        Y_hat = hmm_viterbi(w, X, possible_tags, transition)
        phi_prime = create_features(X, Y_prime)
        phi_hat = create_features(X, Y_hat)
        update(w, phi_prime, phi_hat)

    with open('hmm_percep.model', 'wb') as f:
        pickle.dump((dict(transition), possible_tags, w), f)


def test_hmm_percep():
    with open('hmm_percep.model', 'rb') as f:
        transition, possible_tags, w = pickle.load(f)
    with open(output_file, 'w', encoding='utf8') as f:
        for words in load_test_data():
            Y_hat = hmm_viterbi(w, words, possible_tags, transition)
            f.write(' '.join(Y_hat) + '\n')


def main():
    train_hmm_percep()
    test_hmm_percep()


if __name__ == '__main__':
    import os
    args = arguments_parse()

    test_file = args.input if args.input else '../../data/wiki-en-test.norm'.replace('/', os.sep)
    # test_file = args.input if args.input else '../../test/05-test-input.txt'.replace('/', os.sep)
    train_file = args.train if args.train else '../../data/wiki-en-train.norm_pos'.replace('/', os.sep)
    # train_file = args.train if args.train else '../../test/05-train-input.txt'.replace('/', os.sep)
    output_file = args.output if args.output else 'answer'

    main()


'''
実行結果 (epoch=5)

$ ../../script/gradepos.pl ../../data/wiki-en-test.pos answer

Accuracy: 87.95% (4013/4563)

Most common mistakes:
NNS --> NN      27
NN --> JJ       23
NN --> NNS      21
JJ --> NN       21
NN --> VBG      19
JJ --> RB       19
NN --> NNP      18
NN --> VB       18
NN --> RB       17
JJ --> VBG      17


通常の生成モデルを用いたもの

perl ..\..\script\gradepos.pl ..\..\data\wiki-en-test.pos my_answer.pos

Accuracy: 90.82% (4144/4563)

Most common mistakes:
NNS --> NN      45
NN --> JJ       27
NNP --> NN      22
JJ --> DT       22
VBN --> NN      12
JJ --> NN       12
NN --> IN       11
NN --> DT       10
NNP --> JJ      8
JJ --> VBN      7
'''
