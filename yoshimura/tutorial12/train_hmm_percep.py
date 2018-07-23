from collections import defaultdict
from tqdm import tqdm
import numpy as np
import dill

epoch = 5
# train_path = '../../test/05-train-input.txt'  # テスト用パス
train_path = '../../data/wiki-en-train.norm_pos' 


def init_ids(X, Y, ids):
    for i in range(len(Y) + 1):
        if i == 0:
            first_tag = '<s>'
        else:
            first_tag = Y[i-1]
        if i == len(Y):
            next_tag = '</s>'
        else:
            next_tag = Y[i]
        ids[f'T {first_tag} {next_tag}']
    for i in range(len(Y)):
        ids[f'E {Y[i]} {X[i]}']


def create_feature(X, Y, ids):
    phi = np.zeros(len(ids))
    for i in range(len(Y) + 1):
        if i == 0:
            first_tag = '<s>'
        else:
            first_tag = Y[i-1]
        if i == len(Y):
            next_tag = '</s>'
        else:
            next_tag = Y[i]
        phi += create_trans(first_tag, next_tag, ids)
    for i in range(len(Y)):
        phi += create_emit(Y[i], X[i], ids)
    return phi


def create_trans(first_tag, next_tag, ids):
    phi = np.zeros(len(ids))
    if f'T {first_tag} {next_tag}' in ids:
        phi[ids[f'T {first_tag} {next_tag}']] = 1
    return phi


def create_emit(y, x, ids):
    phi = np.zeros(len(ids))
    if f'E {y} {x}' in ids:
        phi[ids[f'E {y} {x}']] = 1
    return phi


def hmm_viterbi(weights, X, ids, possible_tags):
    '''素性を使ったビタビアルゴリズム'''
    best_score = {}
    best_edge = {}
    best_score['0 <s>'] = 0
    best_edge['0 <s>'] = None

    len_X = len(X)

    for i in range(len_X):
        for prev in possible_tags.keys():
            if f'{i} {prev}' in best_score:
                for next_ in possible_tags.keys():
                    if f'T {prev} {next_}' in ids:
                        score = best_score[f'{i} {prev}'] + np.dot(weights, create_trans(prev, next_, ids) + create_emit(next_, X[i], ids))
                        if f'{i + 1} {next_}' not in best_score or best_score[f'{i + 1} {next_}'] < score:
                            best_score[f'{i + 1} {next_}'] = score
                            best_edge[f'{i + 1} {next_}'] = f'{i} {prev}'
    # </s>に対して同じ操作をを行う
    for tag in possible_tags.keys():
        if f'{len_X} {tag}' in best_score and f'T {tag} </s>' in ids:
            score = best_score[f'{len_X} {tag}'] + np.dot(weights, create_trans(tag, '</s>', ids))
            if f'{len_X + 1} </s>' not in best_score or best_score[f'{len_X + 1} </s>'] < score:
                best_score[f'{len_X + 1} </s>'] = score
                best_edge[f'{len_X + 1} </s>'] = f'{len_X} {tag}'

    # 後ろ向き
    tags = []
    next_edge = best_edge[f'{len_X + 1} </s>']
    while next_edge != '0 <s>':
        # このエッジの品詞を出力に追加
        position, tag = next_edge.split(' ')
        tags.append(tag)
        next_edge = best_edge[next_edge]

    return tags[::-1]

if __name__ == '__main__':
    ids = defaultdict(lambda: len(ids))
    possible_tags = {'<s>': 1, '</s>': 1}
    data = []
    # ファイルを読み込んでデータを作成
    for line in open(train_path, 'r'):
        X = []
        Y = []
        word_tags = line.rstrip('\n').split(' ')
        for word_tag in word_tags:
            word, tag = word_tag.split('_')
            possible_tags[tag] = 1  # 可能なタグとして保存
            X.append(word)
            Y.append(tag)
        data.append((X, Y))
        init_ids(X, Y, ids)

    weights = np.zeros(len(ids))
        
    # 学習
    for _ in tqdm(range(epoch)):
        for X, Y_prime in tqdm(data):
            Y_hat = hmm_viterbi(weights, X, ids, possible_tags)
            phi_prime = create_feature(X, Y_prime, ids)
            phi_hat = create_feature(X, Y_hat, ids)
            weights += phi_prime - phi_hat

    # 保存
    with open('weights', 'wb') as f:
        dill.dump(weights, f)
    with open('ids', 'wb') as f:
        dill.dump(ids, f)
    with open('p_tags', 'wb') as f:
        dill.dump(possible_tags, f)
