import unittest
import nn
import data
import numpy as np

class TestAct(unittest.TestCase):
    def test_0(self):
        s = nn.Sigmoid()
        inputs = np.asarray([0], dtype = np.float16)
        s.call(inputs)
        self.assertEqual(inputs[0], 0.5)
        inputs[:] = 0
        t = nn.Tanh()
        t.call(inputs)
        self.assertEqual(inputs[0], 0)

    def test_pos(self):
        s = nn.Sigmoid()
        inputs = np.arange(11, dtype = np.float)
        s.call(inputs)
        n = np.sum(inputs < 0.9)
        self.assertEqual(n, 3)

        inputs[:] = np.arange(11) - 5
        t = nn.Tanh()
        t.call(inputs)
        for i in range(6):
            self.assertAlmostEqual(inputs[5+i] + inputs[5-i], 0)

    def test_sm0(self):
        s = nn.Softmax()
        inputs = np.asarray([0], dtype = np.float16)
        s.call(inputs)
        self.assertEqual(inputs[0], 1)

class TestEmbedding(unittest.TestCase):
    def __init__(self, *largs, **dargs):
        super(TestEmbedding, self).__init__(*largs, **dargs)
        self.e = nn.EmbeddingLayer(2, 3, None)

    def test_io(self):
        # shape = batch_size, seq_len, id
        inputs = [[0, 1, 0], [1, 0, 1]]
        inputs = np.asarray(inputs, dtype = np.int16)
        outputs = np.empty((2, 3, 3))
        flags = np.asarray([3,3])
        self.e.forward(inputs, outputs, flags)
        self.assertTrue(np.all(outputs[0, 0] == outputs[0, 2]))
        self.assertTrue(np.all(outputs[1, 0] == outputs[1, 2]))

class TestSGD(unittest.TestCase):
    def __init__(self, *largs, **dargs):
        super(TestSGD, self).__init__(*largs, **dargs)
        self.opt = nn.SteepestGradientOptimizer(1)

    def test_run(self):
        w = np.empty((3,3))
        u = np.ones_like(w)
        self.opt.register((w, u))
        self.opt.init_weights()
        for _, e in zip(range(3), (1, 1.9, 2.71)):
            self.opt()
            self.assertEqual(np.sum(u == 0), 9)
            self.assertEqual(np.sum(w == e), 9)

class TestFFL(unittest.TestCase):
    def __init__(self, *largs, **dargs):
        super(TestFFL, self).__init__(*largs, **dargs)
        S = nn.Sigmoid
        T = nn.Tanh
        self.opt = nn.SteepestGradientOptimizer(0.1, 0)
        self.fs = nn.FeedForwardLayer(3, 2, S, self.opt)
        self.ft = nn.FeedForwardLayer(3, 2, T, self.opt)
        self.fx = nn.FeedForwardLayer(2, 1, None, self.opt)
        self.opt.init_weights()

    def test_fw(self):
        i = np.ones((5,3))
        o = np.empty((5,2))
        f = np.ones(5, dtype = np.int)
        self.fs.forward(i, o, f)
        self.assertEqual(np.sum(0.5 == o), 10)

        self.ft.forward(i, o, f)
        self.assertEqual(np.sum(0 == o), 10)

    def test_fwt(self):
        i = np.ones((5, 4, 3))
        o = np.empty((5, 4, 2))
        f = np.ones(5, dtype = np.int) * 4
        f[2] -= 2 # kill 2 units of dim 2
        self.fs.forward(i, o, f)
        self.assertLessEqual(np.sum(0.5 == o), 5*4*2)
        self.ft.forward(i, o, f)
        self.assertLessEqual(np.sum(0 == o), 5*4*2)

    def test_bw(self):
        # x + y + z = 3 -> +1 = 3 -> -1
        i = np.array([[1, 1, 1], [3, -1, 1], [-1, -3, 1], [0, 0, -3], [1, 2, 0]], dtype = np.float)
        c = np.array([[1, -1], [1, -1], [-1, 1], [-1, 1], [1, -1]], dtype = np.float)
        o = np.empty_like(c)
        last_e = np.ones_like(c)
        f = np.ones(5, dtype = np.int)
        self.ft.forward(i, o, f)
        self.assertEqual(np.sum(0 == o), 10)
        for _ in range(10):
            e = c - o
            self.assertEqual(np.sum(e**2 <= last_e**2), 10)
            last_e[:] = e
            self.ft.backward(i, o, e, f)
            self.opt()
            self.ft.forward(i, o, f)

    def test_iterchain(self):
        fw = ['dataset.x', 'l1', 100, 'l2', 200]
        for a, b, c in nn.iter_chain(fw):
            print(a, b, c)
        bw = [(i, i) if isinstance(i, int) else i for i in reversed(fw)]
        for a, b, c in nn.iter_chain(bw):
            print(a, b, c)

    def test_chain(self):
        i = np.ones((3,3))
        h = np.empty((3,2))
        o = np.empty((3,1))
        f = np.ones(3, dtype = np.int)
        self.opt.init_weights()
        self.fs.forward(i, h, f)
        self.fx.forward(h, o, f)
        self.assertEqual(np.sum(0 == o), 3)

class TestRNN(unittest.TestCase):
    def __init__(self, *largs, **dargs):
        super(TestRNN, self).__init__(*largs, **dargs)
        S = nn.Sigmoid
        T = nn.Tanh
        self.opt = nn.SteepestGradientOptimizer(0.1, 0)
        self.rnn = nn.RecurrentLayer(2, 2, None, self.opt)
        self.opt.init_weights()

        i = []; o = [];
        i.append([[1, 1],  [0, 0], [-1, 1], [0, 0]])
        o.append([[0, 0],  [1, 1], [0, 0], [-1, 1]])
        i.append([[-2, 0], [0, 0], [-1, 1], [0, 0]])
        o.append([[0, 0], [-1, 1], [0, 0], [-2, 0]])
        i = np.array(i)
        o = np.array(o)
        y = np.zeros_like(o, dtype = np.float)
        f = np.ones(2, dtype = np.int) * 4
        self.data = i, o, y, f

    def test_fw(self):
        i, o, y, f = self.data
        self.rnn.forward(i, y, f)
        self.assertEqual(np.sum(y == 0), 2 * 4 * 2)

    def test_bw(self):
        i, o, y, f = self.data
        c = o - y
        self.rnn.backward(i, y, c, f)

class TestFFNN(unittest.TestCase):
    def __init__(self, *largs, **dargs):
        super(TestFFNN, self).__init__(*largs, **dargs)
        bow = data.S1DataSet('label_sent.txt', 2, True)
        F = nn.FeedForwardLayer
        S = nn.Sigmoid
        T = nn.Tanh
        opt = nn.SteepestGradientOptimizer(1, 0.9)
        layers = ((F, 2, S), (F, None, T))
        self.w = nn.Network(layers, bow, opt)
        opt.init_weights(np.random.uniform)
        self.opt = opt

    def test_str(self):
        s = str(self.w)
        print(s)
        self.assertEqual(len(s.split('\n')), 7)

    def test_train(self):
        nn.turnoff_tqdm()
        self.opt.regularization(1, 0.01)
        self.w.train(50)
        self.opt.show()

if __name__ == '__main__':
    unittest.main()
