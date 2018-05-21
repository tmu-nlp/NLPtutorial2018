import unittest
import n_gram


class TestFunctions(unittest.TestCase):

    def test_witten_bell(self):
        bigram_count = {('Tottori', 'is'):2, ('Tottori', 'city'):1}
        self.assertEqual(n_gram.witten_bell_weights(bigram_count)['Tottori'], 0.6)

    def test_n_gram(self):
        self.assertEqual(n_gram.n_gram(2, "abc"), (('a', 'b'), ('b', 'c')))

    def test_interpolating(self):
        fun = n_gram.unigram_smooth_gen(0.95, 1/100)
        self.assertAlmostEqual(fun(0), 0.05 * 1/100, delta=0.001) # underflow
        self.assertAlmostEqual(fun(1), 0.95, delta=0.001)

if __name__ == "__main__":
    unittest.main()
