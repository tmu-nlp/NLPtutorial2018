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
            x (numpy.array): size of (batch_size, input_size).
        '''
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        '''
        Args:
            dout (numpy.array): size of (batch_size, input_size).
        '''
        dx = dout * (1.0 - self.out) * self.out

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
        self.x = None   # for calculating grad

    def forward(self, x):
        '''
        Args:
            x (numpy.array): size of (batch_size, input_size).
        '''
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out
    
    def backward(self, dout):
        '''
        Args:
            dout (numpy.array): size of (batch_size, output_size).
        '''
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
        self.t = None
        self.y = None
    
    def forward(self, x, t):
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
        self.t = t
        self.y = softmax(x)
        
        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout):
        '''
        Args:
            dout (float)
        '''
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
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
    print(*['affine:', x, out, dx])
    print(*(['affine(params)'] + affine.params))
    print(*(['affine(grad)'] + affine.grads))

    y = np.array([1]).reshape(1,-1)
    print(*['softmax:', x, y])

    sm = SoftmaxLayer()
    out = sm.forward(x, y)
    dx = sm.backward(out)
    print(*['softmax:', x, out, dx])

    
