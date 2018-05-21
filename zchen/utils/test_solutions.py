from solutions import Trie, any_ranges
import unittest
from n_gram import nlog_gen, unigram_smooth_gen

class DummyModel:
    dummy_data = (
        (('ab',), 0.25),
        (('abc',), 0.25),
        (('acb',), 0.25),
        (('a',), 0.25),
    )

    def set_a(self, tf: bool):
        self._a = tf

    @property
    def iterprob(self):
        if self._a:
            return iter(DummyModel.dummy_data)
        else:
            return iter(DummyModel.dummy_data[:-1])


class TestTrie(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTrie, self).__init__(*args, **kwargs)

        model = DummyModel()
        model.set_a(False)
        us = unigram_smooth_gen(0.95, 1/1000000)
        _nlog = nlog_gen()
        self._trie = Trie(model, us, _nlog)

        eow = Trie.eow()
        value = _nlog(us(0.25))
        data = { 'a': { # eow: value,
                       'b': { eow: value,
                             'c': { eow: value }, },
                       'c': {
                           'b': { eow: value } }
                       }
                }
        self._data = data
        self.assertEqual(self._trie.data, self._data)

    def test_search(self):
        trie = self._trie
        ret = trie.search_through('abce')
        self.assertEqual(list(ret), ['ab', 'abc'])

    def test_range(self):
        for i,j in any_ranges(range(10)):
            self.assertGreater(j, i)
        for i,j in any_ranges([2,5,7]):
            self.assertGreater(j, i)


if __name__ == '__main__':
    unittest.main()
