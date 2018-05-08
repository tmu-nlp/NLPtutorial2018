import sys
sys.path.append("..")
from utils import N_Gram_Family
import unittest

answer = '''1-gram model based on 8 tokens in 5 types
c   0.125000
d   0.125000
a   0.250000
b   0.250000
</s>0.250000
----
2-gram model based on 8 tokens in 6 types
<s>, a0.250000
b, c 0.500000
b, d 0.500000
a, b 1.000000
c, </s>1.000000
d, </s>1.000000
'''

class TestStringMethods(unittest.TestCase):

    def test_train(self):
        model = N_Gram_Family(2, "dummy")
        model.seal_model("../../test/02-train-input.txt")
        self.assertEqual(str(model), answer)

    def test_load(self):
        model = N_Gram_Family(2, "dummy")
        model.load()
        self.assertEqual(str(model), answer)

    def test_entropy(self):
        model = N_Gram_Family(2, "dummy")
        model.load()
        self.assertEqual(model.entropy_of("../../test/02-train-input.txt", [0.95, 0.95], 1000000), -1.93958023291905)

if __name__ == "__main__":
    unittest.main()
