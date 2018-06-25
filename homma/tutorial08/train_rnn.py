# pylint: disable=E1101
import argparse
import os
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm


class Layer(object):
    """RNNの層の基底クラス

    Attributes
    ----------
    out_size : int
        出力のサイズ（Vocab数 or ノード数 or POS_type）
    back : Layer
        前の層
    xt : List(ndarray, size(1, hidden_size))
        各時刻の層の値のリスト
    """

    def __init__(self, out_size, back):
        self.out_size = out_size
        self.back = back
        self.xt = []


class InputLayer(Layer):
    """RNNの入力層 各時刻の入力値を保持しておく
    """

    def __init__(self, in_size, back):
        super.__init__(self, in_size, back)

    def forward(self, x):
        self.xt.append(x)
        return x


class HiddenLayer(Layer):
    """RNNの隠れ層

    Attributes
    ----------
    w_x : ndarray, size(hidden_size, input_size)
        前方向にかかる重み
    w_r : ndarray, size(hidden_size, hidden_size)
        再帰にかかる重み
    b : ndarray, size(1, hidden_size)
        バイアス
    dw_x : ndarray, size(hidden_size, input_size)
        前方向にかかる重みの勾配
    dw_r : ndarray, size(hidden_size, hidden_size)
        再帰にかかる重みの勾配
    db : ndarray, size(1, hidden_size)
        バイアスの勾配
    """

    def __init__(self, hid_size, back):
        super.__init__(self, hid_size, back)
        self.w_x = np.random.uniform(-.1, .1, (hid_size, back.out_size))
        self.w_r = np.random.uniform(-.1, .1, (hid_size, hid_size))
        self.b = np.random.uniform(-.1, .1, (1, hid_size))
        self.dw_x = np.zeros_like(self.w_x)
        self.dw_r = np.zeros_like(self.w_r)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        if self.xt:
            h = np.tanh(np.dot(self.xt[-1], self.w_r) + np.dot(self.w_x, x) + self.b)
        else:
            h = np.tanh(np.dot(self.w_x, x) + self.b)
        self.xt.append(h)
        return h

    def backward(self, delta_r):


    def initialize_deltas(self):
        self.dw_x = np.zeros_like(self.w_x)
        self.dw_r = np.zeros_like(self.w_r)
        self.db = np.zeros_like(self.b)
        if self.back:
            self.back.initialize_deltas()


class OutputLayer(Layer):
    """RNNの出力層

    Attributes
    ----------
    w : ndarray, size(POS_Tpye, hidden_size)
        前方向にかかる重み
    b : ndarray
        出力のバイアス
    dw : ndarray, size(POS_Tpye, hidden_size)
        前方向にかかる重みの勾配
    db : ndarray
        出力のバイアスの勾配
    """

    def __init__(self, out_size, back):
        super.__init__(self, out_size, back)
        self.w = np.random.uniform(-.1, .1, (back.out_size, out_size))
        self.b = np.random.uniform(-.1, .1, (1, out_size))
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        p = np.tanh(np.dot(x, self.w) + self.b)
        self.xt.append(p)

    def backward(self, p_, t):
        print('Output backward')
        err_o = p_ - self.xt[t]
        print(err_o)
        self.dw += np.outer(self.back.xt[t], err_o)
        print(self.dw)
        self.db += err_o
        print(self.db)
        self.back.backward()

    def answer(self, ids):
        ans = []
        for p in self.xt:
            # print(p)
            pass

    def initialize_deltas(self):
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        self.back.initialize_deltas()


class Rnn(object):
    """品詞推定を行うRNNのモデル

    Attributes
    ----------
        train_file : str
            学習ファイルのパス
        model_file : str
            学習結果を保存するファイルのパス
        lam : float
            学習率(0~1)
        epo : int
            エポック，学習回数
        hid : int
            隠れ層のノード数
        lay : int
            隠れ層の数
        ids_x : defaultdict(int)
            離散値の入力単語を連続値のIDに変換する辞書
        ids_y : defaultdict(int)
            離散値の出力品詞を連続値のIDに変換する辞書
        x_size : int
            学習ファイルに含まれている入力単語の種類数
        y_size : int
            学習ファイルに含まれている品詞の種類数
        fea_lab : List
            学習ファイルの単語列と品詞列のペアのリスト
        model : List(int)
            モデルの形（[input, hidden_1_hid, ..., hidden_lay_hid]）
    """

    def __init__(
            self,
            train_file='../../data/wiki-en-train.norm_pos'.replace(
                '/', os.sep),
            model_file='rnn_model',
            lam=0.005,
            epo=3,
            hid=2,
            lay=1,
            model=None):
        self.train_file = train_file
        self.model_file = model_file
        self.lam = lam
        self.epo = epo
        self.hid = hid
        self.lay = lay

        self.ids_x = defaultdict(lambda: len(self.ids_x))
        self.ids_y = defaultdict(lambda: len(self.ids_y))
        self.x_size = 0
        self.y_size = 0
        self.fea_lab = []
        if not model:
            self.model = [0] + [hid for _ in range(lay)] + [0]
        else:
            self.model = model
            # layをmodelにあわせる（-2は入力層と出力層の分の数）
            self.lay = len(model) - 2

    def train(self):
        """学習
        """

        self.read_set_ids()
        self.create_set_feature_labels()

        # 各層を生成し，モデルを作成する
        inp = InputLayer(self.model[0], None)
        hids = [HiddenLayer(self.model[1], inp)]
        for size in self.model[2:-1]:
            hids.append(HiddenLayer(size, hids[-1]))
        out = OutputLayer(self.model[-1], hids[-1])

        for e in tqdm(range(self.epo)):
            for x_list, y_list in self.fea_lab:
                self.forward(x_list, inp, hids, out)
                self.gradient(y_list, out)
            #     update_weights(hid, out)
            out.update_weights() # 出力層から連鎖
            # with open(f'model_{e+1}', 'wb') as f:
            #     pickle.dump(hid, f)
            #     pickle.dump(out, f)
            out.answer(self.ids_y)

    def read_set_ids(self):
        """学習ファイルを読み，idsとそのサイズをセットする
        """
        for line in open(self.train_file, encoding='utf8'):
            for word_pos in line.split():
                word, pos = word_pos.split('_')
                self.ids_x[word]
                self.ids_y[pos]
        self.x_size = len(self.ids_x)
        self.y_size = len(self.ids_y)
        self.model[0] = self.x_size
        self.model[-1] = self.y_size

    def create_set_feature_labels(self):
        """学習ファイルから[各行の素性と正解ラベルのペア]のリストを作成する
        """
        for line in open(self.train_file, encoding='utf8'):
            word_list = []
            pos_list = []
            for word_pos in line.split():
                word, pos = word_pos.split('_')
                word_list.append(
                    np.array(np.eye(self.x_size)[self.ids_x[word]]))
                pos_list.append(np.array(np.eye(self.y_size)[self.ids_y[pos]]))
            self.fea_lab.append((word_list, pos_list))

    def forward(self, x_list, inp, hids, out):
        for x in x_list:
            h = inp.forward(x)
            for hid in hids:
                h = hid.forward(h)
            out.forward(h)

    def gradient(self, y_list, hids, out):
        out.initialize_deltas() # 出力層から連鎖

        for t_, y in enumerate(y_list[::-1]):
            t = len(y_list) - t_ - 1 # 最終単語入力時刻 → 0
            out.backward(y, t) # 出力層から連鎖


if __name__ == '__main__':
    # このファイルがおいてあるディレクトリに移動
    # os.chdir(os.path.dirname(__file__))

    rnn = Rnn(train_file=r'..\..\test\05-train-input.txt')

    rnn.train()
