import numpy as np
import dill

# テストデータファイルのパス
test_path = '../../data/wiki-en-test.norm'
# test_path = '../../test/05-test-input.txt'  # テスト用


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


def forward_rnn(net, words):
    vocab_type = len(words)
    h = [np.ndarray for _ in range(vocab_type)]  # 隠れ層の値（各時間tにおいて）
    p = [np.ndarray for _ in range(vocab_type)]  # 出力の確率分布の値（各時間tにおいて）
    tags_predict = [np.ndarray for _ in range(vocab_type)]  # 出力の確率分布の値(各t)
    # h = np.zeros(vocab_type)  # 隠れ層の値（各時間tにおいて）
    # p = np.zeros(vocab_type)  # 出力の確率分布の値（各時間tにおいて）
    # tags_predict = np.zeros(vocab_type)  # 出力の確率分布の値(各t)
    
    w_ri, b_r, w_rh, w_oh, b_o = net
    
    for t in range(vocab_type):
        if t > 0:
            h[t] = np.tanh(np.dot(w_ri, words[t]) + np.dot(w_rh, h[t-1]) + b_r)
        else:
            h[t] = np.tanh(np.dot(w_ri, words[t]) + b_r)
        p[t] = np.tanh(np.dot(w_oh, h[t]) + b_o)
        tags_predict[t] = find_max(p[t])

    return (h, p, tags_predict)


if __name__ == '__main__':
    # モデル読み込み
    f_net = open('network', 'rb')
    f_word_ids = open('word_ids', 'rb')
    f_tag_ids = open('tag_ids', 'rb')
    with f_net, f_word_ids, f_tag_ids:
        net = dill.load(f_net)
        word_ids = dill.load(f_word_ids)
        tag_ids = dill.load(f_tag_ids)

    # テスト
    with open(test_path, 'r') as test_file:
        for line in test_file:
            words = []
            line = line.rstrip()
            for word in line.split(' '):
                if word in word_ids:
                    words.append(create_one_hot(word_ids[word], len(word_ids)))
                else:
                    words.append(np.zeros(len(word_ids)))

            h, p, tags_predict = forward_rnn(net, words)

            for tag in tags_predict:
                for key, value in tag_ids.items():
                    if value == tag:
                        print(key, end=' ')
            print()
