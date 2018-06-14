import sys, os
sys.path.append(os.pardir)
from common.layers import *

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
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):)
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def forward(self, x, y):
        score = self.predict(x)
        loss = self.last_layer.forward(o, y)
        return loss
    
    def backward(self, dout=1):
        dout = self.last_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

if __name__ == '__main__':
    network = SimpleNeuralNetwork(2, 10, 2)

