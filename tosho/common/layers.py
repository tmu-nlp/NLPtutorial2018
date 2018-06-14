import numpy as np
import sys, os
sys.path.append(os.pardir)
from common.functions import softmax, cross_entropy_error

class AddLayer:
    '''
    加算を行うレイヤー
    '''
    def __init__(self):
        pass
    
    def forward(self, x, y):
        out = x + y

        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy

class MulLayer:
    '''
    積算を行うレイヤー
    '''
    def __init__(self):
        self.x, self.y = None, None
    
    def forward(self, x, y):
        self.x, self.y = x, y
        out = x * y

        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

class SigmoidLayer:
    '''
    Sigmoid関数を作用させるレイヤー
    '''
    def __init__(self):
        self.out = None
    
    def forward(self, x):
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
        self.W = W
        self.b = b
        self.x = None   # for calculate dW
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.W, x) + self.b

        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx

class SoftmaxLayer:
    '''
    softmax変換を行うレイヤー
    '''
    def __init__(self):
        self.y = None
        self.o = None
        self.loss = None
    
    def forward(self, x, y):
        self.y = y
        self.o = softmax(x)
        self.loss = cross_entropy_error(self.o, self.y)

        return self.loss

    def backward(self, dout):
        batch_size = self.y.shape[0]
        if self.y.size == self.o.size:
            dx = (self.o - self.y) / batch_size
        else:
            dx = self.o.copy()
            dx[np.arange(batch_size), self.o] -= 1
            dx = dx / batch_size
        
        return dx

if __name__ == '__main__':
    x, y = 0.2, 0.8

    add = AddLayer()
    out = add.forward(5, 7)
    dx, dy = add.backward(1)
    print(*['add:', x, y, out, dx, dy])

    mul = MulLayer()
    out = mul.forward(x, y)
    dx, dy = mul.backward(1)
    print(*['mul:', x, y, out, dx, dy])

    sig = SigmoidLayer()
    out = sig.forward(np.array([x, y]))
    dx = sig.backward(np.array([1, 1]))
    print(*['sigmoid:', x, y, out, dx])

    W = np.random.rand(3, 2)
    b = np.random.rand(3)
    affine = AffineLayer(W, b)
    out = affine.forward(np.array([x, y]))
    dx = affine.backward(np.array([1, 1]))
    print(*['sigmoid:', W, b])
    print(*['sigmoid:', x, y, out, dx])
    print(*['sigmoid:', affine.dW, affine.db])

    sm = SoftmaxLayer()
    out = sm.forward(np.array([x, y]), np.array([0, 1]))
    dx = sm.backward(np.array([0.2, -0.2]))
    print(*['softmax:', x, y, out, dx])

    
