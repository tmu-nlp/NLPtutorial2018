import sys, os
sys.path.append(os.pardir)
from common.layers import *
from random import randint
import pickle
from itertools import zip_longest

class SGD:
    '''
    Stochastic Gradient Descent
    '''
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]

class Momentum:
    def __init__(self, lr=0.01, mementum=0.9):
        self.lr = lr
        self.momentum = mementum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))

        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += self.v[i]

class Trainer:
    def __init__(self, model, optimizer, model_dir='./model'):
        self.model = model
        self.optimizer = optimizer
        self.model_dir = model_dir

    def train(self, x, y, max_epoch=10, batch_size=32):
        model, optimizer = self.model, self.optimizer

        x_dev, y_dev = x[-1*len(x)//10:], y[-1*len(x)//10:]
        x, y = x[:-1*len(x)//10], y[:-1*len(x)//10]
        data_size = len(x)

        for epoch in range(max_epoch):
            rand_idx = np.random.permutation(np.arange(data_size))
            x, y = x[rand_idx], y[rand_idx]
            loss_list = []
            max_iters = data_size // batch_size - 1
            eval_interval = max(max_iters // 10, 1)

            for i in range(max_iters):
                batch_x = x[i * batch_size : (i + 1) * batch_size]
                batch_y = y[i * batch_size : (i + 1) * batch_size]

                loss = model.forward(batch_x, batch_y)
                model.backward()
                loss_list.append(loss)

                params, grads = self.merge_dups(model.params, model.grads)
                optimizer.update(params, grads)

                if i % eval_interval == 0:
                    print(f'epoch #{epoch+1} | {i+1}/{max_iters} | loss = {loss}')

            acc = self.model.accuracy(x_dev, y_dev)
            print(f'epoch #{epoch+1} | loss = {sum(loss_list)/len(loss_list)} | dev acc = {acc}')

            file_name = f'{self.model_dir}/model_{epoch+1}_{acc*100:.3f}.pkl'
            self.model.save_params(file_name)

    def merge_dups(self, params, grads):
        params, grads = params[:], grads[:]

        while True:
            merged = False
            L = len(params)

            for i in range(0, L-1):
                for j in range(i+1, L):
                    if params[i] is params[j]:
                        grads[i] += grads[j]
                        merged = True
                        params.pop(j)
                        grads.pop(j)
                    if merged: break
                if merged: break
            if not merged: break

        return params, grads

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.I, self.H, self.O = input_size, hidden_size, output_size

        W1 = 0.01 * np.random.randn(self.I, self.H)
        b1 = np.zeros(self.H)
        W2 = 0.01 * np.random.randn(self.H, self.O)
        b2 = np.zeros(self.O)

        self.layers = [
            AffineLayer(W1, b1),
            TanhLayer(),
            AffineLayer(W2, b2)
        ]
        self.last_layer = SoftmaxLayer()

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def save_params(self, file_name='model.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump((self.I, self.H, self.O, self.params), f)

    @staticmethod
    def load_params(model_file='model.pkl'):
        with open(model_file, 'rb') as f:
            input_size, hidden_size, output_size, params = pickle.load(f)

            model = SimpleNeuralNetwork(input_size, hidden_size, output_size)
            for i, p in enumerate(params):
                model.params[i][...] = p

            return model

    def predict(self, x):
        '''
        Args:
            x (numpy.array): size of (batch_size, input_size).
        '''
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def forward(self, x, y):
        '''
        Args:
            x (numpy.array): size of (batch_size, input_size).
            y (numpy.array): size of (batch_size, 1).
                Each element in y represents the id of corresponding answer to x.
        Return:
            float: result of loss function
        '''
        score = self.predict(x)
        loss = self.last_layer.forward(score, y)
        return loss

    def backward(self, dout=1):
        '''
        Args:
            dout (float): loss
        '''
        dout = self.last_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def accuracy(self, x, y):
        o = self.predict(x)
        o = np.argmax(o, axis=1)
        return np.sum(o == y) / len(o)

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.I, self.H, self.O = input_size, hidden_size, output_size

        Wx1 = 0.01 * np.random.randn(self.I, self.H)
        Wh1 = 0.01 * np.random.randn(self.H, self.H)
        b1 = np.zeros(self.H)
        W2 = 0.01 * np.random.randn(self.H, self.O)
        b2 = np.zeros(self.O)

        # Wx1 = np.ones_like(Wx1) / np.sqrt(self.I)
        # Wh1 = np.ones_like(Wh1) / np.sqrt(self.H)
        # b1 = np.zeros(self.H)
        # W2 = np.ones_like(W2) / np.sqrt(self.H)
        # b2 = np.zeros(self.O)

        self.layers = [
            RecurrentLayer(Wx1, Wh1, b1),
            AffineLayer(W2, b2)
        ]
        self.last_layer = SoftmaxLayer()

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

        self.L = []
        self.data_size = []

    def save_params(self, file_name='model.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump((self.I, self.H, self.O, self.params), f)

    @staticmethod
    def load_params(model_file='model.pkl'):
        with open(model_file, 'rb') as f:
            input_size, hidden_size, output_size, params = pickle.load(f)

            model = SimpleRNN(input_size, hidden_size, output_size)
            for i, p in enumerate(params):
                model.params[i][...] = p

            return model

    def predict(self, x, s=0):
        '''
        Args:
            x (numpy.array): size of (batch_size, input_size).
        '''
        for layer in self.layers:
            x = layer.forward(x, s)

        return x

    def forward(self, X, Y):
        '''
        Args:
            x (numpy.array): size of (batch_size, seq_size, input_size).
            y (numpy.array): size of (batch_size, seq_size, 1).
                Each element in y represents the id of corresponding answer to x.
        Return:
            float: result of loss function
        '''
        # (batch_size, seq_size, data_size) => (max(seq_size), batch_size, data_size)
        X = np.array(list(zip_longest(*X, fillvalue=np.zeros_like(X[0][0]))))
        Y = np.array(list(zip_longest(*Y, fillvalue=-1)))

        self.L = []
        for s, x, y in zip(range(len(X)), X, Y):
            score = self.predict(x, s)
            loss = self.last_layer.forward(score, y, s)
            self.L.append(loss)
            self.data_size.append(np.sum(y != -1))
        return sum(self.L) / len(self.L)

    def backward(self, dout=1):
        '''
        Args:
            dout (float): loss
        '''

        # 勾配を初期化
        self.grads = [np.zeros_like(g) for g in self.grads]

        total_data_size = sum(self.data_size)
        for _ in range(len(self.L)):
            dout = self.last_layer.backward(self.data_size.pop()/total_data_size)
            grads = []
            for layer in reversed(self.layers):
                dout = layer.backward(dout)
                grads += reversed(layer.grads)
            for i, g in enumerate(reversed(grads)):
                self.grads[i] += g

        return dout

    def accuracy(self, X, Y):
        # (batch_size, seq_size, data_size) => (max(seq_size), batch_size, data_size)
        X = np.array(list(zip_longest(*X, fillvalue=np.zeros_like(X[0][0]))))
        Y = np.array(list(zip_longest(*Y, fillvalue=-1)))

        T = 0
        L = 0
        for s, x, y in zip(range(len(X)), X, Y):
            o = self.predict(x, s)
            o = np.argmax(o, axis=1)
            mask = (y != -1)
            T += np.sum((o == y) * mask)
            L += np.sum(mask)
            np.sum(o == y) / len(o)

        return T/L

if __name__ == '__main__':
    mode = sys.argv.pop().lower()
    from dataset import logic_circuit

    if mode == 'load':
        model = SimpleRNN(5000,256,40)
        model.save_params()
        model = SimpleRNN.load_params()

    if mode == 'nn':
        x_train, t_train = logic_circuit.load_data(operant='AND', data_size=50000)
        print(x_train.shape)
        print(t_train.shape)

        model = SimpleNeuralNetwork(x_train.shape[1], 2, 2)
        optimizer = SGD(0.1)

        trainer = Trainer(model, optimizer)

        print(f'Initial | ACC: {model.accuracy(x_train, t_train)}')

        trainer.train(x_train, t_train, max_epoch=20, batch_size=100)
        print(f'Trained | ACC: {model.accuracy(x_train, t_train)}')

    if mode == 'rnn':
        seq_size = 4

        x_train, t_train = logic_circuit.load_data(operant='AND', data_size=500)
        x_train = x_train.reshape(500 // seq_size, seq_size, 2)
        t_train = t_train.reshape(500 // seq_size, seq_size,)
        print(x_train.shape)
        print(t_train.shape)

        model = SimpleRNN(2, 3, 2)
        optimizer = SGD(0.1)

        trainer = Trainer(model, optimizer)

        for p in model.params: print(p)

        print(f'Initial | ACC: {model.accuracy(x_train, t_train)}')

        trainer.train(x_train, t_train, max_epoch=1, batch_size=25)

        print(f'Trained | ACC: {model.accuracy(x_train, t_train)}')

        for p in [p for l in model.layers for p in l.params]: print(p)

        model.save_params()

        # x = [
        #     [[1, 1], [1, 1]],
        #     [[1, 1], [0, 0]]
        # ]
        # y = [
        #     [1, 1],
        #     [1, -1]
        # ]

        # x, y = np.array(x), np.array(y)
        # def print_iter(it):
        #     for i in it:
        #         print(i)

        # model = SimpleRNN(2, 4, 3)
        # optimizer = SGD(lr=0.01)

        # print('initial params')
        # print_iter(model.params)
        # print('initial grads')
        # print_iter(model.grads)

        # loss = model.forward(x, y)
        # model.backward()
        # print(loss)

        # optimizer.update(model.params, model.grads)

        # print('updated params')
        # print_iter(model.params)
        # print('updated grads')
        # print_iter(model.grads)
