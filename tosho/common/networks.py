import sys, os
sys.path.append(os.pardir)
from common.layers import *
from random import randint

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
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train(self, x, y, max_epoch=10, batch_size=32):
        data_size = len(x)
        model, optimizer = self.model, self.optimizer
        
        for epoch in range(max_epoch):
            print(f'epoch #{epoch}', end='')

            rand_idx = np.random.permutation(np.arange(data_size))
            x, y = x[rand_idx], y[rand_idx]
            loss_list = []

            for i in range(data_size // batch_size):
                batch_x = x[i * batch_size : (i + 1) * batch_size]
                batch_y = y[i * batch_size : (i + 1) * batch_size]

                loss = model.forward(batch_x, batch_y)
                model.backward()
                loss_list.append(loss)

                params, grads = self.merge_dups(model.params, model.grads)
                optimizer.update(params, grads)
            
            print(f' | loss = {sum(loss_list)/len(loss_list)}')
    
    def merge_dups(self, params, grads):
        params, grads = params[:], grads[:]

        while True:
            merged = False
            L = len(params)

            for i in range(0, L-1):
                for j in range(i+1, L):
                    if params[i] is params[j]:
                        params[i] += params[j]
                        merged = True
                        params.pop(j)
                        grads.pop(j)
                    
                    if merged: break
                if merged: break
            if not merged: break
        
        return params, grads

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, H)
        b2 = np.zeros(H)

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

if __name__ == '__main__':
    from dataset import logic_circuit

    x_train, t_train = logic_circuit.load_data(operant='AND', data_size=50000)
    print(x_train.shape)
    print(t_train.shape)

    model = SimpleNeuralNetwork(x_train.shape[1], 2, 2)
    optimizer = SGD(0.1)
    
    trainer = Trainer(model, optimizer)

    print(f'Initial | ACC: {model.accuracy(x_train, t_train)}')
    # for p in model.params: print(p)

    trainer.train(x_train, t_train, max_epoch=20, batch_size=100)
    print(f'Trained | ACC: {model.accuracy(x_train, t_train)}')
    # for p in model.params: print(p)