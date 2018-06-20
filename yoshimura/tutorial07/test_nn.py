# from train_nn.py import forward_nn
import numpy as np
import dill
　
# テストファイルのパス
test_path = '../../data/titles-en-test.word'
# test_path = 'test.txt'  # テスト用


def create_features(sentence, ids):
    '''未知語は無視してsentenceの素性を作成する'''
    phi = np.zeros(len(ids))
    words = sentence.split(' ')
    # 1-gram素性
    for word in words:
        if 'UNI:' + word in ids:
            phi[ids['UNI:' + word]] += 1
    return phi


def forward_nn(net, phi_zero):
    phis = [phi_zero]  # 各層の値
    for i in range(len(net)):
        w, b = net[i]
        # 前の層に基づいて値を計算
        phis.append(np.tanh(np.dot(w, phis[i]) + b))
    return phis  # 各層の結果を返す


if __name__ == '__main__':
    # モデル読み込み
    with open('network', 'rb') as f_net, open('ids', 'rb') as f_ids:
        net = dill.load(f_net)
        ids = dill.load(f_ids)

    # テスト
    with open(test_path, 'r') as test_file:
        for line in test_file:
            line = line.rstrip()
            phi_zero = create_features(line, ids)
            score = forward_nn(net, phi_zero)[len(net)]
            if score > 0:
                predict = 1
            else:
                predict = -1
            print(predict)
