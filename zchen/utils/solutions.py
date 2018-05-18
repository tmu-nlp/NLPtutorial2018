import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def grid_search(model, testfile, delta = 100):
    weight_range = np.linspace(.01, .99, delta)
    idx_uni, idx_bi = np.meshgrid(weight_range, weight_range)
    entropy = np.zeros((delta, delta), dtype = np.float)
    for uni_i, uni in tqdm(enumerate(weight_range), total = delta):
        for bi_i, bi in enumerate(weight_range):
            model.prepare([uni, bi], 1/1000000)
            entropy[uni_i, bi_i] = model.entropy_of(testfile)
    return idx_uni, idx_bi, entropy


def arg_range(tuple_of_ndarray):
    c = tuple_of_ndarray[:-1]
    Z = tuple_of_ndarray[-1]
    idx_Z_min = np.unravel_index(np.argmin(Z), Z.shape)
    idx_Z_max = np.unravel_index(np.argmax(Z), Z.shape)
    return tuple(i[idx_Z_min] for i in c) + (Z[idx_Z_min],), tuple(i[idx_Z_max] for i in c) + (Z[idx_Z_max],)


def any_ranges(ordered):
    length = len(ordered)
    if length < 2:
        raise StopIteration("not a good range")
    for i in range(length-1):
        for j in range(i+1, length):
            yield ordered[i], ordered[j]


class Trie:
    """
    Our trie node implementation. Very basic. but does the job
    """
    _eow  = 'vAlUe'

    @classmethod
    def eow(cls):
        return cls._eow

    def __init__(self, model, interpolator, trans_prob = lambda x:x):
        unk_term = interpolator(0)
        self._unk_value = trans_prob(unk_term)
        self._data = {}
        self._cache = {}
        for ng, value in model.iterprob:
            curr = self._data
            word = ' '.join(ng)
            for char in word:
                if char not in curr:
                    curr[char] = {}
                    curr = curr[char]
                else:
                    curr = curr[char]
            curr[Trie._eow] = trans_prob(interpolator(value))

    def search_through(self, word: str):# -> bool:
        curr = self._data
        for i, char in enumerate(word):
            if char in curr:
                curr = curr[char]
                if Trie._eow in curr:
                    subw = word[:i+1]
                    self._cache[subw] = curr[Trie._eow]
                    yield subw

    def __getitem__(self, word: str):
        return self._cache[word]

    def set_unk(self, word: str):
        #print("DBGjl", word, self._cache, word in self._cache , self._cache[word] != self._unk_value)
        if word in self._cache and self._cache[word] != self._unk_value or word == '':
            raise ValueError("Never happen in set_unk for '%s'" % word)
        # print("set unk", word)
        self._cache[word] = self._unk_value

    @property
    def unk_value(self):
        return self._unk_value

    @property
    def data(self):
        return self._data


def viterbi(model: Trie, tokens: str, verbose = False):
    def forward():
        def step_back_from(stratum):
            if stratum['end']:
                path_loss_gen = ((tok, model[tok]+strata[start - len(tok)]['min_loss']) for tok in stratum['end'])
                tok, min_loss = min(path_loss_gen, key = lambda x:x[1])
                stratum['solution'] = tok
                stratum['based_on'] = start - len(tok)
                stratum['min_loss'] = min_loss
        length = len(tokens)
        strata = [defaultdict(set) for i in range(length + 1)]
        strata[0]['min_loss'] = 0
        coverage = []
        unk_start = length
        for start in range(length): # the last one will be left behind
            # find edge from start to all already ends
            fractions = []
            word_gen = model.search_through(tokens[start:])
            for i, word in enumerate(word_gen):
                end = start + len(word)
                strata[end]['end'].add(word)
                fractions.append(end)
                # print(length, start, strata[start],fractions)
                # if word == 'は':
                #     print(start, word, 'in dict')

            # guarantee for continuity
            if not fractions or fractions[0] - start > 1:
                char = tokens[start]
                strata[start+1]['end'].add(char)
                # print(start, char, fractions, char)
                model.set_unk(char)
                if verbose and fractions and unk_start == length: # to link continuous unks
                    unk_start = start
                # continue # is bad for continuity

            # face fractions
            elif verbose and fractions:
                # of one prefix
                if unk_start < start - 1 < length: # link continuous unks
                    frac_word = tokens[unk_start:start-1]
                    strata[end]['end'].add(frac_word)
                    model.set_unk(frac_word)
                    unk_start = length

                # and many suffixes
                for i,j in any_ranges(fractions):
                    frac_word = tokens[i:j]
                    strata[j]['end'].add(frac_word)
                    print("frac:", frac_word)
                    try:
                        next(model.search_through(frac_word))
                    except StopIteration:
                        model.set_unk(frac_word)
                for s in strata:
                    print(s)
                print('\n\n')

            # bode for step with in 0:start
            step_back_from(strata[start])
        start += 1
        step_back_from(strata[length])
        return strata

    def backward(strata):
        based_on = -1
        solution = []
        while based_on:
            # print("back", strata[based_on]['solution'], 'to', based_on)
            solution.append(strata[based_on]['solution'])
            based_on = strata[based_on]['based_on']
        solution.reverse()
        return solution

    return backward(forward())


def plot_2d_contour(x_y_entropy):
    X, Y, Z = x_y_entropy
    plt.figure()
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Grid Search')
    plt.show()
