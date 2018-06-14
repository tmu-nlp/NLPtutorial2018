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
        self.out = None
    
    def forward(self, x):
        '''
        Args:
            x (numpy.array) : size of (batch_size, dimension).
        '''
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

class AffineLayer:
    '''
     線形変換を行うレイヤー
    '''
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None   # for calculating grad

    def forward(self, x):
        '''
        Args:
            x (numpy.array) : size of (batch_size, dimension).
        '''
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out
    
    def backward(self, dout):
        W, b = self.params

        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
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
        self.y = None
        self.o = None
    
    def forward(self, x, y):
        '''
        This function will perform softmax on x and 
        return the loss of its result compared with y

        Args:
            x (numpy.array) : size of (batch_size, dimension).
            y (numpy.array) : size of (batch_size, 1).
                Each element in y represents the id of corresponding answer to x.
        Return:
            float: result of cross_entropy_error
        '''
        self.y = y
        self.o = softmax(x)
        
        loss = cross_entropy_error(self.o, self.y)
        return loss

    def backward(self, dout):
        batch_size = self.y.shape[0]

        dx = self.o.copy()
        dx[np.arange(batch_size), self.y] -= 1
        dx *= dout
        dx = dx / batch_size
        
        return dx

if __name__ == '__main__':
    x = np.random.randn(1, 5)

    sig = SigmoidLayer()
    out = sig.forward(x)
    dx = sig.backward(out)
    print(*['sigmoid:', x, out, dx])

    W = np.random.rand(5, 10)
    b = np.random.rand(10)
    affine = AffineLayer(W, b)
    out = affine.forward(x)
    dx = affine.backward(out)
    print(*['affine:', x, *affine.params, out, dx])
    print(*affine.grads)

    y = np.array([1]).reshape(1,-1)
    print(*['softmax:', x, y])

    sm = SoftmaxLayer()
    out = sm.forward(x, y)
    dx = sm.backward(out)
    print(*['softmax:', x, out, dx])

    
