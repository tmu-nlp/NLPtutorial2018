import unittest
import nn
import numpy as np

class TestSigmoid(unittest.TestCase):
    def test_0(self):
        s = nn.Sigmoid()
        inputs = np.asarray([0], dtype = np.float16)
        s(inputs)
        self.assertEqual(inputs[0], 0.5)

    def test_pos(self):
        s = nn.Sigmoid()
        inputs = np.arange(10, dtype = np.float16)
        s(inputs)
        n = np.sum(inputs < 0.9)
        self.assertEqual(n, 3)

    def test_sm0(self):
        s = nn.Softmax()
        inputs = np.asarray([0], dtype = np.float16)
        s(inputs)
        self.assertEqual(inputs[0], 1)

class TestEmbedding(unittest.TestCase):
    def __init__(self, *largs, **dargs):
        super(TestEmbedding, self).__init__(*largs, **dargs)
        self.e = nn.EmbeddingLayer(2, 3, None)

    def test_feed(self):
        # shape = batch_size, seq_len, id
        inputs = [[0, 1, 0], [1, 0, 1]]
        inputs = np.asarray(inputs, dtype = np.int16)
        outputs = self.e.feed(inputs)
        self.assertTrue(np.all(outputs[0, 0] == outputs[0, 2]))
        self.assertTrue(np.all(outputs[1, 0] == outputs[1, 2]))

class TestFFNN:
    pass

if __name__ == '__main__':
    unittest.main()
