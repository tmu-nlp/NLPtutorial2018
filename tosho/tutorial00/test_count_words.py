import unittest
import numpy as np
from count_words import count_words

class TestCountWords(unittest.TestCase):
    def test_count_words(self):
        target_file = "./../../test/00-input.txt"
        actual = count_words(target_file)

        expected = {
            "a": 1,
            "b": 2,
            "c": 2,
            "d": 1
        }

        self.assertEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()