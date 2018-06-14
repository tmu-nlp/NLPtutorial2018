import sys, os
sys.path.append(os.pardir)
from common.layers import *
from random import randint

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        self.layers = [
            AffineLayer(W1, b1),
            SigmoidLayer(),
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

if __name__ == '__main__':
    I, H, O = 5, 10, 2
    batch_size = 3
    
    x = np.random.randn(batch_size, 5)
    y = np.array([randint(0, O-1) for _ in range(batch_size)]).reshape(batch_size,-1)

    network = SimpleNeuralNetwork(I, H, O)
    
    o = network.predict(x)
    loss = network.forward(x, y)
    dout = network.backward(loss)

    for pair in zip(x, y, o, dout):
        print(*pair)
    
    for pair in zip(network.params, network.grads):
        print(*pair)