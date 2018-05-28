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
    _eow  = 'vAlUe'

    @classmethod
    def eow(cls):
        return cls._eow

    def __init__(self, model, interpolator, trans_prob = lambda x:x):
        unk_term = interpolator(0)
        self._unk_value = trans_prob(unk_term)
        self._data = {}
        self._cache = {}
        for ng, value in model.iter_prob:
            curr = self._data
            word = ng[-1] # I know how to implement bi-gram version
            value = trans_prob(interpolator(value))
            for char in word:
                if char not in curr:
                    curr[char] = {}
                    curr = curr[char]
                else:
                    curr = curr[char]
            curr[Trie._eow] = value

    def search_through(self, word: str):
        curr = self._data
        for i, char in enumerate(word):
            if char in curr:
                curr = curr[char]
                if Trie._eow in curr:
                    subw = word[:i+1]
                    self._cache[subw] = curr[Trie._eow]
                    yield subw
            else:# bloody bug: need more negative test case
                break

    def __getitem__(self, word: str):
        return self._cache[word]

    def set_unk(self, word: str, warning_len = 10):
        #print("DBGjl", word, self._cache, word in self._cache , self._cache[word] != self._unk_value)
        if word in self._cache and self._cache[word] != self._unk_value or word == '':
            raise ValueError("Never happen in set_unk for '%s'" % word)
        # print("set unk", word)
        if len(word) > warning_len:
            raise ValueError("see", word)
        self._cache[word] = self._unk_value

    @property
    def unk_value(self):
        return self._unk_value

    @property
    def data(self):
        return self._data


def viterbi(model: Trie, tokens: str, **verbose):
    def forward():
        def step_back_from(start):
            stratum = strata[start]
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
        verbose_str = ''
        for start in range(length): # the last one will be left behind
            # [discovery]
            fractions = []
            word_gen = model.search_through(tokens[start:])
            for word in word_gen:
                end = start + len(word)
                strata[end]['end'].add(word)
                fractions.append(end)
                if verbose and len(word) > verbose.get('warning_len', 10):
                    verbose_str += "Add long word at pos(%d) '%s';\n" % (start, word)

            # [fractions] in discovery
            if verbose and verbose.get('fragmentize', False):
                if len(fractions) > verbose.get('fraction_size', 5):
                    verbose_str += "Find %d fractions at pos(%d);\n" % (len(fractions), start)
                for i,j in any_ranges(fractions):
                    frac_word = tokens[i:j]
                    strata[j]['end'].add(frac_word)

                    last_result = None
                    for last_result in model.search_through(frac_word):
                        pass
                    if last_result != frac_word:
                        model.set_unk(frac_word)

            # [unk] guarantee for continuity of chars
            if not fractions or fractions[0] - start > 1:
                char = tokens[start]
                strata[start+1]['end'].add(char)
                model.set_unk(char)
                if not fractions and unk_start == length: # [long unk]
                    unk_start = start
                    if verbose:
                        verbose_str += "Find unk (possible longer) '%s' start at pos(%d)\n" % (char, start)

            # [long unk] JIT
            elif fractions:
                frac_word = tokens[unk_start:start]
                if unk_start < start - 2 < length:
                    strata[start]['end'].add(frac_word) # Once costy bug: python name scope 'start'
                    model.set_unk(frac_word)
                    if verbose:
                        verbose_str += "Finish long unk '%s'(%d,%d)\n" % (frac_word, unk_start, start)
                elif verbose:
                    verbose_str += "Cancel long unk '%s'(%d,%d)\n" % (frac_word, unk_start, start)
                unk_start = length

            step_back_from(start)
        step_back_from(length)
        return strata, verbose_str

    def backward(strata):
        based_on = -1
        solution = []
        while based_on:
            # print("back", strata[based_on]['solution'], 'to', based_on)
            solution.append(strata[based_on]['solution'])
            based_on = strata[based_on]['based_on']
        solution.reverse()
        return solution

    strata, verbose_str = forward()
    if verbose:
        for i, s in enumerate(strata):
            verbose_str += f"{i} {s}\n"
        verbose_str += "\n"
        return backward(strata), verbose_str
    return backward(strata)


def plot_2d_contour(x_y_entropy):
    X, Y, Z = x_y_entropy
    plt.figure()
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Grid Search')
    plt.show()
