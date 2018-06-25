import numpy as np
import sys
sys.path.append("..")
from utils.n_gram import n_gram
from utils.data import load_train_dataset, split_dataset

class Sigmoid:
    def __call__(self, inputs):
        return 1 / (np.exp(-inputs) + 1)

    def grad(self, outputs, errors):
        return outputs * (1 - outputs) * errors

class SteepestGradientOptimizer:
    def __init__(self, weights, learning_rate, momentum = 0.9):
        self._ws = weights
        self._lr = learning_rate
        self._mt = momentum
        self._mm = None

    def __call__(self, updates):
        if self._mt:
            if self._mm is None:
                self._mm = updates.copy()
            else:
                np.add((1-self._mt) * updates, self._mt * self._mm, out = self._mm)
        self._ws += self._lr * (self._mm if self._mt else updates)

class FeedForwardLayer:
    def __init__(self, in_dim, out_dim, act, opt):
        self._weights = np.zeros((out_dim, in_dim))
        self._updates = np.zeros_like(self._weights)
        self._biases = np.zeros((out_dim, 1)) # broadcast
        self._io = None
        self._errors = None
        self._act = act
        self._opt = opt

    def feed(self, inputs, prev_errors = None):
        self._batch_size = batch_size = inputs.shape[1]
        if self._io:
            # in_place
            if self._io[0].shape[0] != inputs.shape[0] or self._io[0].shape[1] > batch_size:
                raise ValueError("feeding input in strange size.")
            self._io[0] = inputs
        else:
            # initialize
            outputs = np.empty((out_dim, batch_size))
            self._io = [inputs, outputs]
            # train mode
            if prev_errors:
                self._prev_errors = prev_errors
                self._errors = errors = np.empty_like(outputs)
                return outputs, errors
            # predict mode
            return outputs

    def forward(self):
        # io.shape == (io_dim, batch_size)
        i, o = self._io
        o = o[:, self._batch_size:]
        np.matmul(self._weights, i, out = o)
        np.add(o, self._biases, out = o)
        self._act(o)

    @property
    def outputs(self):
        return self._io[1][:, self._batch_size]

    def backward(self, errors):
        # prepare
        prev_errors = self._prev_errors
        i, o = self._io
        batch_size = self._batch_size
        if errors.shape[1] < batch_size:
            i = i[:, :batch_size]
            o[:, batch_size:] = 0
            o = o[:, :batch_size]
            prev_errors[:, batch_size:] = 0
            prev_errors = prev_errors[:, self._batch_size:]
        # errors.shape == (out_dim, batch_size)
        self._act.grad(o, errors)
        np.matmul(self._weights.T, errors, out = prev_errors)
        # (o, i) = (o, b) * (b, i)
        np.matmul(errors, i.T, out = self._updates)
        self._opt(self._updates)

class Network:
    def __init__(self, layer_shape_act = ((30, None), (100, Sigmoid), (10, Sigmoid)), opt):
        layers = []
        for (i, _), (o, a) in n_gram(2, layer_shape_act):
            layers.append(FeedForwardLayer(i, o, a, opt))
        self._layers = layers

    def train(self, epochs, train_set, valid_set):
        # build connections
        fisrt_batch = train_set.dummy_x
        prev_errors = None
        for layer in self._layers:
            first_batch, prev_errors = layer.feed(fisrt_batch, prev_errors)

        best_model = None
        # train with train_set
        for epoch in range(epochs):
            # train
            for batch in train_set:
                # forward
                self._layers[0].feed(batch.x)
                for layer in self._layers:
                    layer.forward()
                errors = self._layers[-1].outputs - batch.y
                # backward errors
                self._layers[-1].backward(errors)

            # validate and save current model
            i, j = 0, 0
            for batch in valid_set:
                self._layers[0].feed(batch.x)
                for layer in self._layers:
                    layer.forward()
                errors = (self._layers[-1].outputs - 0.5 > 0) != batch.y
                i += np.sum(errors)
                j += len(errors)
            if best_model is None or best_model[0] > i/j:
                summary = [layer.weights for layers in self._layers]
                best_model = i/j, summary
        # choose the best model with valid_set performance
        # return the best model


def DataSet:
    def __init__(self, fname, batch_size):
        self._X = []
        self._Y = []
        vocab = defaultdict(lambda w:len(vocab))
        for x, y in load_train_dataset(vocab, fname):
            self._X.append(x)
            self._Y.append(y)
        self._vocab = vocab
        self._batch_size = batch_size
        self._reader = 0

    def shuffle(self):
        pass

    def __iter__(self):
        # make numpy batch
        i = self._reader
        x = self._X[i:i+self._batch_size]
        y = self._Y[i:i+self._batch_size]
        x = _vectorize(self._vocab, x)
        y = np.asarray(y)
        db = DataBatch(x, y)
        self._reader += self._batch_size
        return db

    @property
    def dummy_x(self):
        x = self._X[:self._batch_size]
        x = _vectorize(self._vocab, x)
        return x


class DataBatch:
    def __init__(self, X, Y):
        self.x = X
        self.y = Y
