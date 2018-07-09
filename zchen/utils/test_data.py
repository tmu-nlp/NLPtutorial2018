import unittest
from data import S1DataSet, SSDataSet

class TestS1DataSet(unittest.TestCase):
    def __init__(self, *largs, **dargs):
        super(TestS1DataSet, self).__init__(*largs, **dargs)
        self.bow = S1DataSet('label_sent.txt', 2, True)
        self.seq = S1DataSet('label_sent.txt', 2, False)

    def test_str(self):
        bow_str = 'Squeezed bow dataset, vocab size: 5\nShape: (2, 5)'
        seq_str = 'Sequential id dataset, vocab size: 5\nShape: (2, 4) Max seq len: 4'
        self.assertEqual(bow_str, str(self.bow))
        self.assertEqual(seq_str, str(self.seq))

    def test_iter(self):
        yhat = self.bow.create_buffer()
        self.assertEqual(yhat.shape, (2,))
        for X, Y, F in self.bow:
            print(X, Y, F)
        yhat = self.seq.create_buffer()
        self.assertEqual(yhat.shape, (2,))
        for X, Y, F in self.seq:
            print(X, Y, F)


class TestSSDataSet(unittest.TestCase):
    def __init__(self, *largs, **dargs):
        super(TestSSDataSet, self).__init__(*largs, **dargs)
        self.seq = SSDataSet('../../test/05-train-input.txt', 3)

    def test_str(self):
        seq_str = 'Sequential id dataset\n tok vocab: 3\n pos vocab: 3\nBatch & Max seq len: (3, 3)\n'
        self.assertEqual(seq_str, str(self.seq))

    def test_iter(self):
        yhat = self.seq.create_buffer()
        for X, Y, F in self.seq:
            print(X, Y, F)

if __name__ == '__main__':
    unittest.main()
