import numpy as np
from n_gram import n_gram, interpolate_gen
from data import split_dataset, S1DataSet
#from multiprocessing import Pool#Process
from tqdm import tqdm

def turnoff_tqdm():
    global tqdm
    def nothing(x, **args):
        return x
    tqdm = nothing

class Sigmoid:
    @staticmethod
    def call(inputs):
        # 1 / ( 1 + e^{-x} )
        np.exp(-inputs, out = inputs)
        np.add(inputs, 1, out = inputs)
        np.divide(1, inputs, out = inputs)

    @staticmethod
    def grad(outputs, errors):
        np.multiply(outputs * (1 - outputs), errors, out = errors)

class Tanh:
    @staticmethod
    def call(inputs):
        # 2 / ( 1 + e^{-2x} ) - 1
        np.exp(-2*inputs, out = inputs)
        np.add(inputs, 1, out = inputs)
        np.divide(2, inputs, out = inputs)
        inputs[:] -= 1

    @staticmethod
    def grad(outputs, errors):
        np.multiply( (1 - outputs ** 2), errors, out = errors)

class Softmax:
    @staticmethod
    def call(inputs):
        Sigmoid.call(inputs)
        n = np.sum(inputs)
        np.divide(inputs, n, out = inputs)

    @staticmethod
    def grad(outputs, errors):
        pass


class SteepestGradientOptimizer:
    def __init__(self, learning_rate, momentum = 0.9):
        self._wu = []
        self._lr = learning_rate
        self._mt = interpolate_gen(momentum) if momentum else None
        self._mu = None
        self._re = None

    def register(self, weights_updates):
        self._wu.append(weights_updates)

    def change_lr(self, ratio = 0.5):
        self._lr *= ratio

    def regularization(self, order, c):
        if 0 < order < 3 and isinstance(order, int) and 0 < c < 1 and isinstance(c, float):
            self._re = (order, c)
        else:
            self._re = None

    def init_weights(self, func = np.zeros):
        for w, _ in self._wu:
            if func is np.zeros:
                w[:] = 0
            elif func is np.ones:
                w[:] = 1
            elif isinstance(func, (int, float)):
                w[:] = func
            else:
                w[:] = func(size = w.shape)

    def __call__(self):
        if self._mt:
            if self._mu is None:
                self._mu = [(u.copy(), u) for _, u in self._wu]
            else:
                for m, u in self._mu:
                    m[:] = self._mt(m, u)
        for i, (w, u) in enumerate(self._wu):
            w += self._lr * (self._mu[i][0] if self._mt else u)
            if self._re:
                order, c = self._re
                if order == 1:
                    w[np.where(np.abs(w) < c)] = 0
                    w -= np.sign(w) * c
                elif order == 2:
                    w *= c
            u[:] = 0

    def show(self):
        for w, u in self._wu:
            print(w)

class Layer:
    def forward(self, inputs, outputs):
        raise NotImplementedError('')

    def backward(self, output, errors, prev_errors):
        raise NotImplementedError('')

class FeedForwardLayer(Layer):
    def __init__(self, in_dim, out_dim, act, opt = None):
        self._weights = np.empty((in_dim, out_dim))
        self._biases = np.empty(out_dim)
        self._act = act if act else None
        if opt: # training model
            self._updates = np.zeros_like(self._weights)
            self._bias_updates = np.zeros_like(self._biases)
            opt.register((self._weights, self._updates))
            opt.register((self._biases,  self._bias_updates))

    def __str__(self):
        s = 'FeedForwardLayer (%d->%d)' % self._weights.shape
        #s += ' with %s'
        return s

    def forward(self, inputs, outputs, flags):

        def mp(i, o, f):
            if len(i.shape) == 1:
                f,t = i.shape + o.shape
            else:
                t = f
            np.matmul(i[:f], self._weights, out = o[:t])
            o[:t] += self._biases
            if self._act:
                self._act.call(o[:t])

        # for FFNN, batch is like time serial
        for i, o, f in zip(inputs, outputs, flags):
            if f:
                # self._pool.apply(mp, args = (i, o, f))
                # also error in Process()
                mp(i, o, f)


    def backward(self, inputs, outputs, errors, flags, prev_errors = None):
        if prev_errors is None:
            prev_errors = [None] * len(flags)

        for i, o, e, f, p in zip(inputs, outputs, errors, flags, prev_errors):
            if f:
                if len(i.shape) == 1:
                    f,t = i.shape + o.shape
                else:
                    t = f
                if self._act:
                    self._act.grad(o[:t], e[:t])
                if p is not None:
                    np.matmul(self._weights, e[:t], out = p[:f])
                # (o, i) = (o, b) * (b, i)
                self._updates += np.outer(i[:f], e[:t])
                self._bias_updates += e[:t]


class EmbeddingLayer(Layer):
    def __init__(self, in_dim, out_dim, opt = None):
        self._emb = np.random.uniform(size = (in_dim, out_dim))
        if opt:
            self._updates = np.zeros_like(self._emb)
            opt.register((self._emb, self._updates))

    def forward(self, inputs, outputs, flags):
        # like FFNN, Embedding is not sensible to timestep or batch
        # batch
        for i, o, f in zip(inputs, outputs, flags):
            # time
            for pos in range(f):
                o[pos] = self._emb[i[pos]]

    def backward(self, inputs, errors, flag):
        for i, o, f in zip(inputs, outputs, flags):
            for pos in range(f):
                self._updates[i[pos]] += errors[pos]

class RecurrentLayer(Layer):
    def __init__(self, in_dim, out_dim, act, opt = None):
        self._square = np.empty((out_dim, out_dim))
        self._in_weights = np.empty((out_dim, in_dim))
        self._biases = np.empty(out_dim)
        self._act = act
        if opt:
            self._sq_updates = np.zeros_like(self._square)
            self._in_updates = np.zeros_like(self._in_weights)
            self._bi_updates = np.zeros_like(self._biases)
            opt.register((self._square, self._sq_updates))
            opt.register((self._in_weights, self._in_updates))
            opt.register((self._biases, self._bi_updates))

    def forward(self, inputs, outputs, flags):
        # batch
        for i,o,f in zip(inputs, outputs, flags):
            # thread
            for pos in range(f):
                np.matmul(i[pos], self._in_weights, out = o[pos])
                if pos:
                    o[pos] += np.matmul(o[pos - 1], self._square)
            o[pos] += self._biases
        if self._act:
            self._act.call(outputs)# care for softmax

    def backward(self, inputs, outputs, errors, flags, prev_errors = None):
        if prev_errors is None:
            prev_errors = [None] * len(flags)

        for i, o, e, f, p in zip(inputs, outputs, errors, flags, prev_errors):
            # single thread
            for pos in reversed(range(f)):
                if self._act:
                    self._act.grad(o[pos], e[pos])
                self._in_updates += np.outer(i[pos], e[pos])
                self._sq_updates += np.outer(o[pos - 1], e[pos])
                self._bi_updates += e[pos]
                if p is not None:
                    np.matmul(self._in_weights, e[pos], out = p[pos])
                if pos:
                    e[pos-1] += np.matmul(e[pos], self._square.T)

def iter_chain(chain):
    n = len(chain)
    for i in range(n//2):
        l = 2*i+1
        yield chain[l-1], chain[l], chain[l+1]

class Network:
    def __init__(self, layer_shape_act, dataset, opt = None):
        # [b t dataset.x] (F, 40, Sigmoid) [b t 40] (R, dataset.y(None), Softmax) [b t dataset.y]
        chain = [dataset]
        for l,s,a in layer_shape_act:
            if s is None:
                chain.append(l(chain[-1].shape[-1], dataset.y_dim, a, opt))
            else:
                chain.append(l(chain[-1].shape[-1], s, a, opt))
            # if s is None, the final layer
            chain.append(dataset.create_buffer(s))
        self._chain = chain
        self._opt = opt

    def __str__(self):
        s = 'Network:\n'
        s += ' (%s)\n' % str(self._chain[0]).replace('\n', ' ')
        for elem in self._chain[1:]:
            if isinstance(elem, np.ndarray):
                s += ' %s\n' % str(elem.shape)
            else:
                s += ' [%s]\n' % str(elem)
        return s

    def train(self, epochs):

        best_model = None
        # train with train_set
        dataset = self._chain[0]
        backward_chain = [(i, np.empty_like(i)) if isinstance(i, np.ndarray) else i for i in reversed(self._chain)]
        # (y, e) [layer] (y, e) [layer] (dataset.x)

        for epoch in tqdm(range(epochs), desc = 'epoch'):
            # train
            mse = 0 # shall be determined by dataset
            for X, Y, F in tqdm(dataset, total = dataset.num_batch, desc = 'batch'):
                # forward
                for dx, layer, dy in iter_chain(self._chain):
                    if dx is dataset:
                        layer.forward(X, dy, F)
                    else:
                        layer.forward(dx, dy, F)
                # errors
                errors = backward_chain[0][1][:] = Y - self._chain[-1]
                errors[np.where(F==0)] = 0
                mse += np.sum(errors**2)
                # print('True Y', Y)
                # print('Outputs:', self._chain[-1])
                # print('Errors:', backward_chain[0])
                # backward errors
                for (dy, ey), layer, dx in iter_chain(backward_chain):
                    if dx is dataset:
                        layer.backward(X, dy, ey, F)
                    else:
                        layer.backward(dx[0], dy, ey, F, dx[1])

                self._opt()


            # validate and save current model
            #continue
            #for X, Y, F in valid_set:
            #    for dx, layer, dy in iter_chain(self._chain):
            #        if dx is dataset:
            #            layer.forward(X, dy, F)
            #        else:
            #            layer.forward(dx, dy, F)
            #acc = valid_set.measure(self._chain[-1])
            #if best_model is None or best_model[0] > acc:
            #    summary = [layer.weights for layers in self._layers]
            #    best_model = acc, summary
            print("Epoch loss(#mse):", mse)
        # choose the best model with valid_set performance
        # return the best model

    def predict(self, with_x):
        pass
