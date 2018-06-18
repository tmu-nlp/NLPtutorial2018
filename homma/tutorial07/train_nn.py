# pylint: disable=E1101
import argparse
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def arguments_parse():
    parser = argparse.ArgumentParser(
        usage='\tNNで学習を行うプログラム',
        description='説明',
        epilog='end',
        add_help=True,
    )
    parser.add_argument('-t', '--train', help='学習用ファイル名', type=str)
    parser.add_argument('-o', '--output', help='出力ファイル名', type=str)
    return parser.parse_args()


def create_features(raw_sentence, ids):
    '''学習データの1行から素性を作成する'''
    phi = np.zeros(len(ids))
    words = raw_sentence.split()
    for word in words:
        phi[ids['UNI:' + word]] += 1
    return phi


def init_get_network(feature_size, layer, node):
    '''ランダムにネットワークを初期化して返す'''
    # 1つ目の隠れ層
    w0 = np.random.rand(node, feature_size) / 5 - 0.1
    b0 = np.random.rand(1, node) / 5 - 0.1
    net = [(w0, b0)]

    # 隠れ層の中間層
    while len(net) < layer:
        w = np.random.rand(node, node) / 5 - 0.1
        b = np.random.rand(1, node) / 5 - 0.1
        net.append((w, b))

    # 出力層
    w_o = np.random.rand(1, node) / 5 - 0.1
    b_o = np.random.rand(1, 1) / 5 - 0.1
    net.append((w_o, b_o))

    return net


def forward_nn(net, phi0):
    phi = [0 for _ in range(len(net) + 1)]
    phi[0] = phi0
    for i in range(len(net)):
        w, b = net[i]
        phi[i + 1] = np.tanh(np.dot(w, phi[i]) + b).T
    return phi


def backword_nn(net, phi, label):
    J = len(net)
    # 各層の誤差
    delta = np.zeros(J + 1, dtype=np.ndarray)
    # 出力層の誤差
    delta[-1] = np.array([label - phi[J][0]])
    # tanhの勾配を考慮した誤差
    delta_p = np.zeros(J + 1, dtype=np.ndarray)

    for i in range(J, 0, -1):
        delta_p[i] = delta[i] * (1 - np.square(phi[i]))
        w, _ = net[i - 1]
        delta[i - 1] = np.dot(delta_p[i], w)

    return delta


def update_weights(net, phi, delta, lambda_):
    '''重みwを更新する'''
    for i in range(len(net)):
        w, b = net[i]
        w += lambda_ * np.outer(delta[i + 1], phi[i])
        b += lambda_ * delta[i + 1]


def train_nn(train_file, output_file, lambda_=0.1, epoch=1, hidden_l=1, hidden_n=2):
    '''NNで学習を行う'''
    # 学習1回、隠れ層1つ,隠れ層のノード2つ
    ids = defaultdict(lambda: len(ids))
    feature_labels = []

    # 学習ファイルを読み込み，素性を数える
    for line in open(train_file, encoding='utf8'):
        _, raw_sentence = line.strip().split('\t')
        for word in raw_sentence.split():
            ids['UNI:' + word]

    # 学習ファイルを読み込み，素性を作る
    for line in open(train_file, encoding='utf8'):
        raw_label, raw_sentence = line.strip().split('\t')
        label = int(raw_label)
        phi = create_features(raw_sentence, ids)
        feature_labels.append((phi, label))

    # ネットワークを初期化
    net = init_get_network(len(ids), hidden_l, hidden_n)

    # 学習
    for _ in tqdm(range(epoch)):
        for phi0, label in feature_labels:
            phi = forward_nn(net, phi0)
            delta = backword_nn(net, phi, label)
            update_weights(net, phi, delta, lambda_)

    # 書き込み
    with open(output_file, 'wb') as f:
        pickle.dump(net, f)
        pickle.dump(dict(ids), f)


if __name__ == '__main__':
    args = arguments_parse()

    train_file = args.train if args.train else r'..\..\data\titles-en-train.labeled'
    model_file = args.output if args.output else r'nn_model'

    train_nn(train_file, model_file, lambda_=0.03, epoch=30, hidden_n=8)
