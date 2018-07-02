import numpy as np
import sys, os
sys.path.append(os.pardir)
from common.functions import softmax, cross_entropy_error

class SigmoidLayer:
    '''
    Sigmoid関数を作用させるレイヤー
    '''
    def __init__(self):
        self.params, self.grads = [], []
        self.out = []

    def forward(self, x, s=0):
        '''
        Args:
            x (numpy.array): size of (batch_size, input_size).
            s (int, default:0): time step of input.
        '''

        if s == 0 and len(self.out) > 0:
            self.out = []

        out = 1 / (1 + np.exp(-x))
        self.out.append(out)

        return out

    def backward(self, dout):
        '''
        Args:
            dout (numpy.array): size of (batch_size, input_size).
        '''
        out = self.out.pop()

        dx = dout * (1.0 - out) * out

        return dx

class AffineLayer:
    '''
     線形変換を行うレイヤー
    '''
    def __init__(self, W, b):
        '''
        Args:
            W (numpy.array): size of (input_size, output_size)
            b (numpy.array): size of (output_size)
        '''
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = []   # for calculating grad

    def forward(self, x, s=0):
        '''
        Args:
            x (numpy.array): size of (batch_size, input_size).
            s (int, default:0): time step of input.
        '''
        W, b = self.params

        if s == 0:
            self.x = []

        out = np.dot(x, W) + b
        self.x.append(x)
        return out

    def backward(self, dout):
        '''
        Args:
            dout (numpy.array): size of (batch_size, output_size).
        '''
        W, b = self.params
        x = self.x.pop()

        dx = np.dot(dout, W.T)
        dW = np.dot(x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx

class SoftmaxLayer:
    '''
    softmax変換を行うレイヤー
    '''
    def __init__(self):
        self.params, self.grads = [], []
        self.t = []
        self.y = []

    def forward(self, x, t, s=0):
        '''
        This function will perform softmax on x and
        return the loss of its result compared with y

        Args:
            x (numpy.array): size of (batch_size, input_size).
            y (numpy.array): size of (batch_size, 1).
                Each element in y represents the id of corresponding answer to x.
        Return:
            float: result of cross_entropy_error
        '''
        if s == 0:
            self.t, self.y = [], []

        y = softmax(x)

        self.t.append(t)
        self.y.append(y)

        loss = cross_entropy_error(y, t)
        return loss

    def backward(self, dout):
        '''
        Args:
            dout (float)
        '''
        t, y = self.t.pop(), self.y.pop()
        mask = (t == -1)
        batch_size = t.shape[0]

        dx = y.copy()
        dx[np.arange(batch_size), t] -= 1
        dx[mask] *= 0
        dx *= dout
        dx = dx / (t.shape[0] - np.sum(mask))

        return dx

class TanhLayer:
    '''
    tanh関数を作用させるレイヤー
    '''
    def __init__(self):
        self.params, self.grads = [], []
        self.out = []

    def forward(self, x, s=0):
        if s == 0:
            self.out = []

        out = np.tanh(x)
        self.out.append(out)

        return out

    def backward(self, dout):
        out = self.out.pop()
        dx = dout * (1 - out**2)
        return dx

class RecurrentLayer:
    def __init__(self, W, Wh, b):
        '''
        Args:
            W (numpy.array): size of (input_size, output_size)
            b (numpy.array): size of (output_size)
        '''
        self.params = [W, Wh, b]
        self.grads = [np.zeros_like(W), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = []
        self.dh = None

    def forward(self, x, s=0):
        '''
        Args:
            x (numpy.array): size of (batch_size, input_size).
            s (int, default:0): time step of input.
        '''
        W, Wh, b = self.params

        if s == 0:
            self.cache = []
            h_prev = np.zeros((x.shape[0], Wh.shape[0]))
        else:
            h_prev = self.cache[-1][2]

        out = np.dot(x, W) + np.dot(h_prev, Wh) + b
        out = np.tanh(out)

        # 使用したパラメータを保存する
        self.cache.append((x, h_prev, out))

        return out

    def backward(self, dout):
        '''
        Args:
            dout (numpy.array): size of (batch_size, output_size).
        '''
        W, Wh, b = self.params
        x, h_prev, h = self.cache.pop()

        if self.dh is not None:
            dout += self.dh

        dt = dout * (1 - h**2)

        db = np.sum(dt, axis=0)
        dWh = np.dot(h_prev.T, dt)
        dh = np.dot(dt, Wh.T)
        dW = np.dot(x.T, dt)
        dx = np.dot(dt, W.T)

        self.grads[0][...] = dW
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        self.dh = dh

        return dx

if __name__ == '__main__':
    x = np.random.randn(2, 5)

    sig = SigmoidLayer()
    out = sig.forward(x)
    dx = sig.backward(out)
    print(*['sigmoid:', x, out, dx])

    W = np.random.rand(5, 10)
    b = np.random.rand(10)
    affine = AffineLayer(W, b)
    out = affine.forward(x)
    dx = affine.backward(out)
    print(*['affine:', x, out, dx])
    print(*(['affine(params)'] + affine.params))
    print(*(['affine(grad)'] + affine.grads))

    y = np.array([1]).reshape(1,-1)
    print(*['softmax:', x, y])

    sm = SoftmaxLayer()
    out = sm.forward(x, y)
    dx = sm.backward(out)
    print(*['softmax:', x, out, dx])

    tanh = TanhLayer()
    out = tanh.forward(x)
    dx = tanh.backward(out)
    print(*['tanh', x, out, dx])

    # forward/backward multi times
    out = [sig.forward(x, s) for s in range(3)]
    dx = [sig.backward(out.pop()) for _ in range(3)]
    print(*['sigmoid(3):', x, out, dx])

    # Recurrent
    Wx = np.random.rand(5, 10)
    Wh = np.random.rand(10, 10)
    b = np.random.rand(10)
    rec = RecurrentLayer(Wx, Wh, b)
    out = []
    for _ in range(3):
        o = softmax(rec.forward(x))
        o[range(len(o)), 1] -= 1
        out.append(o)
        print(out[-1])
    for dx in out[::-1]:
        print(rec.backward(dx))
        print(rec.dh)