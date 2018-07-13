import random
from collections import defaultdict, Counter
import numpy as np

def load_text(fname, delim = ' '):
    with open(fname, 'r') as fr:
        for line in fr:
            yield line.strip().split(delim)

def load_label_text(fname = '../../data/titles-en-train.labeled'):
    with open(fname) as fr:
        for line in fr:
            label, sentence = line.split('\t') # tokenize & feature extraction
            yield int(label), sentence.strip().split()

def load_label_text_vocab(fname = '../../data/titles-en-train.labeled'):
    data = []; labels = []
    vocab = defaultdict(lambda: len(vocab))
    for label, text in load_label_text(fname):
        labels.append(label)
        data.append(tuple(vocab[w] for w in text))
    return labels, data, vocab

def load_o_s_gen(fname = "../../data/wiki-en-train.norm_pos"):
    with open(fname) as fr:
        for line in fr:
            line = line.strip().split()
            yield (o_s.split('_') for o_s in line)

def load_tok_pos_vocab(fname):
    tok_vocab = defaultdict(lambda: len(tok_vocab))
    pos_vocab = defaultdict(lambda: len(pos_vocab))
    T = []; P = []; maxlen = 0
    for t_p_gen in load_o_s_gen(fname):
        tok_idx = []; pos_idx = []; length = 0
        for t, p in t_p_gen:
            tok_idx.append(tok_vocab[t])
            pos_idx.append(pos_vocab[p])
            length += 1
        if length > maxlen:
            maxlen = length
        T.append(tok_idx)
        P.append(pos_idx)
    return T, P, tok_vocab, pos_vocab, maxlen


def split_dataset(inputs, outputs, train_set_ratio, shuffle = True):
    total = len(inputs)
    idx = list(range(total))
    if shuffle: random.shuffle(idx)
    train_set_ratio = int(total * train_set_ratio)
    train_set_idx = idx[:train_set_ratio]
    valid_set_idx = idx[train_set_ratio:]
    validation_x = tuple(inputs [i] for i in valid_set_idx)
    validation_y = tuple(outputs[i] for i in valid_set_idx)
    return total, train_set_idx, (validation_x, validation_y)

class SSDataSet:
    def __init__(self, fname, batch_size):
        x, y, x_vocab, y_vocab, max_len = load_tok_pos_vocab(fname)
        self._XY = x, y
        self._vocabs = x_vocab, y_vocab
        self._batch_flags = np.empty(batch_size, dtype = np.uint)
        self._maxlen = max_len
        self._buffs = []

    def __str__(self):
        tvs, pvs = (len(v) for v in self._vocabs)
        s = 'Sequential id dataset\n'
        s += ' tok vocab: %d\n' % tvs
        s += ' pos vocab: %d\n' % pvs
        s += 'Batch & Max seq len: %s\n' % str(self.shape)
        return s

    def create_buffer(self, size = None):
        if size is None:
            x = np.empty(self.shape, dtype = np.uint)
            y = np.empty(self.shape, dtype = np.uint)
            self._x_y_buff = x, y
            yhat = np.empty_like(y)
            self._buffs.append(yhat)
            return yhat

        b = len(self._batch_flags)
        buff = np.empty((b, self._maxlen, size))
        self._buffs.append(buff)
        return buff

    @property
    def y_dim(self):
        return self._maxlen

    @property
    def shape(self):
        b = len(self._batch_flags)
        return (b, self._maxlen)

    def eval(self, Yhats):
        #np.sign(Yhats) == Y
        #i += np.sum(errors)
        #j += len(errors)
        pass

    def __iter__(self):
        self._reader = 0
        return self

    def __next__(self):
        # make numpy batch
        i = self._reader
        if i < 0:
            raise StopIteration

        F = self._batch_flags
        batch_size = len(F)
        X, Y = self._x_y_buff
        _X, _Y = self._XY
        _X = _X[i:i+batch_size]
        _Y = _Y[i:i+batch_size]
        current_bs = len(_X)
        F[current_bs:] = 0
        F[:current_bs] = [len(_x) for _x in _X]
        for x, _x, y, _y, f in zip(X, _X, Y, _Y, F):
            x[:f] = _x
            y[:f] = _y
        if current_bs < batch_size:
            self._reader = -1
        else:
            self._reader += batch_size
        return X, Y, F

class S1DataSet:
    def __init__(self, fname, batch_size, squeeze_bow):
        max_len = 0
        y, x, vocab = load_label_text_vocab(fname)
        if squeeze_bow:
            x = [Counter(i) for i in x]
        else:
            max_len = max(len(i) for i in x)
        self._Y = y
        self._X = x
        self._vocab = vocab
        self._batch_flags = np.empty(batch_size, dtype = np.uint)
        self._maxlen = max_len
        self._buffs = []
        self._num_batch = len(x) // batch_size
        if len(x) % batch_size:
            self._num_batch += 1

    def __str__(self):
        s = 'Sequential id' if self._maxlen else 'Squeezed bow'
        s += ' dataset, vocab size: %d\n' % len(self._vocab)
        s += 'Shape: %s' % str(self.shape)
        if self._maxlen:
            s += ' Max seq len: %d' % self._maxlen
        return s

    def create_buffer(self, size = None):
        if size is None:
            b = len(self._batch_flags)
            x = np.empty(self.shape, dtype = np.uint)
            y = np.empty((b, 1), dtype = np.int8)
            self._x_y_buff = x, y
            yhat = np.empty_like(y, dtype = np.float)
            self._buffs.append(yhat)
            return yhat

        b, _maxlen = self.shape
        if self._maxlen:
            buff = np.empty((b, _maxlen, size))
        else:
            buff = np.empty((b, size))
        self._buffs.append(buff)
        return buff

    @property
    def num_batch(self):
        return self._num_batch

    @property
    def y_dim(self):
        return 1

    @property
    def shape(self):
        b = len(self._batch_flags)
        if self._maxlen:
            return (b, self._maxlen)
        else:
            return (b, len(self._vocab))

    def eval(self, Yhats):
        #np.sign(Yhats) == Y
        #i += np.sum(errors)
        #j += len(errors)
        pass

    def holdout(self, ratio, shuffle = False):
        total, train_set_idx, (validation_x, validation_y) = split_dataset(self._X, self._Y, ratio, shuffle)
        self._X = [self._X[i] for i in train_set_idx]
        self._Y = [self._Y[i] for i in train_set_idx]

    def __iter__(self):
        self._reader = 0
        return self

    def __next__(self):
        # make numpy batch
        i = self._reader
        if i < 0:
            raise StopIteration

        F = self._batch_flags
        batch_size, seqlen = self.shape
        X, Y = self._x_y_buff
        _X = self._X[i:i+batch_size]
        _Y = self._Y[i:i+batch_size]
        current_bs = len(_X)
        F[current_bs:] = 0
        if self._maxlen:
            F[:current_bs] = [len(_x) for _x in _X]
            for x, _x, f in zip(X, _X, F):
                x[:f] = _x
        else:
            F[:current_bs] = seqlen
            X[:current_bs] = 0
            for i, batch in enumerate(_X):
                for j,c in batch.items():
                    X[i,j] = c
        Y[:current_bs] = np.array(_Y).reshape(current_bs, 1)

        if current_bs < batch_size:
            self._reader = -1
        else:
            self._reader += batch_size
        return X, Y, F
