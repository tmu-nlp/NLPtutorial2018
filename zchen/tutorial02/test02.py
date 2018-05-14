import sys
sys.path.append("..")
from utils.n_gram import N_Gram_Family
import unittest

answer = '''1-gram model based on 8 tokens in 5 types
c     0.125000
d     0.125000
a     0.250000
b     0.250000
</s>  0.250000
----
2-gram model based on 8 tokens in 6 types with Witten-Bell weights
<s>, a   0.250000
b, c     0.500000
b, d     0.500000
a, b     1.000000
c, </s>  1.000000
d, </s>  1.000000
	- Witten Bell weights:
	w'<s>'          0.666667
	w'a'            0.666667
	w'b'            0.500000
	w'c'            0.500000
	w'd'            0.500000
'''
answer_3 = answer + \
'''----
3-gram model based on 8 tokens in 6 types with Witten-Bell weights
<s>, <s>, a  0.250000
a, b, c      0.500000
a, b, d      0.500000
<s>, a, b    1.000000
b, c, </s>   1.000000
b, d, </s>   1.000000
	- Witten Bell weights:
	w'<s>'          0.666667
	w'a'            0.500000
	w'b'            0.500000
'''

class TestNGramMethods(unittest.TestCase):

    def test_train(self):
        model = N_Gram_Family(2, "dummy")
        model.build("../../test/02-train-input.txt")
        model.seal()
        self.assertEqual(str(model), answer)

    def test_load(self):
        model = N_Gram_Family(2, "dummy")
        model.load()
        model.seal()
        self.assertEqual(str(model), answer)

    def test_entropy(self):
        model = N_Gram_Family(2, "dummy")
        model.load()
        model.seal()
        model.prepare([0.95, 0.95], 1/1000000)
        self.assertEqual(model.entropy_of("../../test/02-train-input.txt"), -1.920456383244844)

    def test_entropy_witten_bell(self):
        model = N_Gram_Family(2, "dummy")
        model.load()
        model.seal()
        model.prepare(None, 1/1000000)
        self.assertEqual(model.entropy_of("../../test/02-train-input.txt"), -2.365348396962379)

    def test_train_3(self):
        model = N_Gram_Family(3, "dummy")
        model.build("../../test/02-train-input.txt")
        model.seal()
        self.assertEqual(str(model), answer_3)

    def test_load_3(self):
        model = N_Gram_Family(3, "dummy")
        model.load()
        model.seal()
        self.assertEqual(str(model), answer_3)

if __name__ == "__main__":
    unittest.main()
