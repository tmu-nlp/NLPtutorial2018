from collections import defaultdict
import numpy as np
import dill
# from tqdm import tqdm

# 学習データファイルのパス
train_path = '../../data/titles-en-train.labeled'
# train_path = '../../test/03-train-input.txt'  # テスト用

# ハイパーパラメータ
epoch = 10
lam = 0.03   # 学習率
layer = 1
hidden_node = 10


def create_features(sentence, ids):
    '''sentenceから素性を作る'''
    phi = np.zeros(len(ids))
    words = sentence.split(' ')
    # 1-gram素性
    for word in words:
        phi[ids['UNI:' + word]] += 1
    return phi


def initialize_net_randomly(ids):
    '''ネットワークをランダムな値で初期化'''
    net = []  # 各層の(w,b)を要素にするリスト

    # 入力層
    w_in = (np.random.rand(hidden_node, len(ids)) - 0.5)/5
    b_in = (np.random.rand(hidden_node) - 0.5)/5
    net.append((w_in, b_in))
    # 隠れ層
    for _ in range(layer - 1):
        w = (np.random.rand(hidden_node, hidden_node) - 0.5)/5
        b = (np.random.rand(hidden_node) - 0.5)/5
        net.append((w, b))
    # 隠れ層
    w_out = (np.random.rand(1, hidden_node) - 0.5)/5
    b_out = (np.random.rand(1) - 0.5)/5
    net.append((w_out, b_out))

    return net


def forward_nn(net, phi_zero):
    phis = [phi_zero]  # 各層の値
    for i in range(len(net)):
        w, b = net[i]
        # 前の層に基づいて値を計算
        phis.append(np.tanh(np.dot(w, phis[i]) + b))
    return phis  # 各層の結果を返す


def backward_nn(net, phis, label):
    J = len(net)
    delta = [np.ndarray for _ in range(J)]
    delta.append(label - phis[J])
    delta_ = [np.ndarray for _ in range(J+1)]
    for i in range(J-1, -1, -1):
        delta_[i+1] = delta[i+1] * (1 - phis[i+1]**2)
        w, b = net[i]
        delta[i] = np.dot(delta_[i+1], w)
    return delta_


def update_weights(net, phis, delta_, lam):
    for i in range(len(net)):
        w, b = net[i]
        w += lam * np.outer(delta_[i+1], phis[i])
        b += lam * delta_[i+1]


if __name__ == '__main__':

    ids = defaultdict(lambda: len(ids))
    feat_label = []  # (phi, label)を要素にするリスト

    # データを読み込んで素性を数える
    with open(train_path, 'r') as f:
        train = []  # (label, sentence)
        for line in f:
            line = line.rstrip()
            label, sentence = line.split('\t')
            train.append((int(label), sentence))
            for word in sentence.split(' '):
                ids['UNI:' + word]

    # 素性作成
    for label, sentence in train:
        feat_label.append((create_features(sentence, ids), label))

    # ネットをランダムに初期化
    net = initialize_net_randomly(ids)

    # 学習を行う
    for _ in range(epoch):
        error = 0
        for phi_zero, label in feat_label:
            phis = forward_nn(net, phi_zero)
            delta_ = backward_nn(net, phis, label)
            update_weights(net, phis, delta_, lam)
            error += abs(label - phis[len(net)][0])
        print(error)

    # モデル書き出し
    with open('network', 'wb') as f_network, open('ids', 'wb') as f_ids:
        dill.dump(net, f_network)
        dill.dump(ids, f_ids)

