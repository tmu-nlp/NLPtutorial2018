from collections import defaultdict
import numpy as np
import dill
from tqdm import tqdm
import ramdom

# ハイパーパラメータ
epoch = 20
hidden_node = 84
lam = 0.006  # 学習率

# 学習データファイルのパス
train_path = '../../data/wiki-en-train.norm_pos'
# train_path = '../../test/05-train-input.txt'  # テスト用


def create_one_hot(id_, size):
    vec = np.zeros(size)
    vec[id_] = 1
    return vec


def find_max(p):
    tag_predict = 0
    for i in range(1, len(p)):
        if p[i] > p[tag_predict]:
            tag_predict = i
    return tag_predict


def initialize_net_randomly(vocab_type, pos_type):
    '''ネットワークをランダムな値で初期化'''

    w_ri = (np.random.rand(hidden_node, vocab_type) - 0.5)/5
    b_r = (np.random.rand(hidden_node) - 0.5)/5
    w_rh = (np.random.rand(hidden_node, hidden_node) - 0.5)/5
    w_oh = (np.random.rand(pos_type, hidden_node) - 0.5)/5
    b_o = (np.random.rand(pos_type) - 0.5)/5

    net = (w_ri, b_r, w_rh, w_oh, b_o)

    return net


def initialize_delta(vocab_type, pos_type):
    '''誤差を初期化'''
    dw_ri = np.zeros((hidden_node, vocab_type))
    dw_rh = np.zeros((hidden_node, hidden_node))
    db_r = np.zeros(hidden_node)
    dw_oh = np.zeros((pos_type, hidden_node))
    db_o = np.zeros(pos_type)

    delta = (dw_ri, dw_rh, db_r, dw_oh, db_o)
    return delta


def forward_rnn(net, words):
    vocab_type = len(words)
    h = [np.ndarray for _ in range(vocab_type)]  # 隠れ層の値（各時間tにおいて）
    p = [np.ndarray for _ in range(vocab_type)]  # 出力の確率分布の値（各時間tにおいて）
    tags_predict = [np.ndarray for _ in range(vocab_type)]  # 出力の確率分布の値(各t)
    
    w_ri, b_r, w_rh, w_oh, b_o = net
    
    for t in tqdm(range(vocab_type), desc='foward'):
        if t > 0:
            h[t] = np.tanh(np.dot(w_ri, words[t]) + np.dot(w_rh, h[t-1]) + b_r)
        else:
            h[t] = np.tanh(np.dot(w_ri, words[t]) + b_r)
        p[t] = np.tanh(np.dot(w_oh, h[t]) + b_o)
        tags_predict[t] = find_max(p[t])

    return (h, p, tags_predict)


def gradient_rnn(net, words, h, p, tags, vocab_type, pos_type):
    delta = initialize_delta(vocab_type, pos_type)
    dw_ri, dw_rh, db_r, dw_oh, db_o = delta
    w_ri, b_r, w_rh, w_oh, b_o = net

    delta_r_ = np.zeros(len(b_r))  # 次の時間から伝搬するエラー
    for t in tqdm(range(len(words))[::-1], desc='gradient'):
        delta_o_ = tags[t] - p[t]  # 出力層エラー
        # 出力層重み勾配
        dw_oh += np.outer(delta_o_, h[t])
        db_o += delta_o_

        if t == len(words) - 1:
            delta_r = np.dot(delta_o_, w_oh)
        else:
            delta_r = np.dot(delta_r_, w_rh) + np.dot(delta_o_, w_oh)  # 逆伝搬
        delta_r_ = delta_r * (1 - h[t]**2)  # tanhの勾配

        # 隠れ層の重み勾配
        dw_ri += np.outer(delta_r_, words[t])
        db_r += delta_r_

        if t != 0:
            dw_rh += np.outer(delta_r_, h[t-1])

    return (dw_ri, dw_rh, db_r, dw_oh, db_o)


def update_weights(net, delta, lam):
    dw_ri, dw_rh, db_r, dw_oh, db_o = delta
    w_ri, b_r, w_rh, w_oh, b_o = net

    w_ri += lam * dw_ri
    w_rh += lam * dw_rh
    b_r += lam * db_r
    w_oh += lam * dw_oh
    b_o += lam * db_o


if __name__ == '__main__':

    word_ids = defaultdict(lambda: len(word_ids))
    tag_ids = defaultdict(lambda: len(tag_ids))

    # 素性を数える
    for line in open(train_path, 'r'):
        line = line.rstrip()
        word_tags = line.split(' ')
        for word_tag in word_tags:
            word, tag = word_tag.split('_')
            word_ids[word]
            tag_ids[tag]
    
    # vocabのsizeとpos_sizeを取得
    vocab_type = len(word_ids)
    pos_type = len(tag_ids)

    # 素性作成
    feat_label = []  # 一文ごとの単語とタグのペア
    for line in open(train_path, 'r'):
        words = []  # 一文の各単語のone-hot
        tags = []  # 一文の各単語のタグのone-hot

        line = line.rstrip()
        word_tags = line.split(' ')
        for word_tag in word_tags:
            word, tag = word_tag.split('_')
            words.append((create_one_hot(word_ids[word], vocab_type)))
            tags.append((create_one_hot(tag_ids[tag], pos_type)))
        feat_label.append((words, tags))

    # ネットワークを初期化
    net = initialize_net_randomly(vocab_type, pos_type)

    # 学習を行う
    for _ in tqdm(range(epoch), desc='epoch'):
        for words, tags in tqdm(feat_label):
            h, p, tags_predict = forward_rnn(net, words)
            delta = gradient_rnn(net, words, h, p, tags, vocab_type, pos_type)
            update_weights(net, delta, lam)

    # モデル書き出し
    f_net = open('network', 'wb')
    f_word_ids = open('word_ids', 'wb')
    f_tag_ids = open('tag_ids', 'wb')
    with f_net, f_word_ids, f_tag_ids:
        dill.dump(net, f_net)
        dill.dump(word_ids, f_word_ids)
        dill.dump(tag_ids, f_tag_ids)

'''
Accuracy 89.31%
Most common mistakes:
Most common mistakes:
JJ --> NN       72
NNS --> NN      68
NNP --> NN      48
VBN --> NN      31
VBG --> NN      19
NN --> JJ       17
IN --> WDT      17
VB --> NN       15
RB --> NN       15
VBP --> NN      14
'''