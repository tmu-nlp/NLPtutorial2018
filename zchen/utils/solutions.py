import matplotlib.pyplot as plt
from .n_gram import N_Gram_Family, nlog_gen
import numpy as np
from tqdm import tqdm


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


class Trie:
    """
    Our trie node implementation. Very basic. but does the job
    """
    _eow  = 'vAlUe'

    def __init__(self, model, interpolator):
        _nlog = nlog_gen()
        oov_term = interpolator(0)
        self._non_key_value = _nlog(oov_term)
        self._data = {}
        self._cache = {}
        self._recent = None
        for ng, value in model.iterprob:
            curr = self._data
            word = ' '.join(ng)
            for char in word:
                if char not in curr:
                    curr[char] = {}
                    curr = curr[char]
                else:
                    curr = curr[char]
            curr[Trie._eow] = _nlog(interpolator(value))

    def possible(self, char: str, idx: int):# -> bool:
        if self.recent is None or idx != self._recent[0]:
            curr = self._data
        else:
            curr = self._recent[1]
        if char not in curr:
            self._recent = None
            return False
        curr = curr[char]
        if self._recent:
            self._recent[1] = curr
            self._recent[2] += char
        else:
            self._recent = [idx, curr, char]
        if Trie._eow in curr:
            self._cache[self._recent[2]] = curr[Trie._eow]
        return True

    @property
    def recent(self):
        ret = self._recent
        if ret and ret[2] in self._cache:
            return ret[2]
        return None

    def __getitem__(self, word: str):
        return self._cache[word]

    def set_non_key_value(self, word: str):
        if word not in self._cache:
            self._cache[word] = self._non_key_value
        elif self._cache[word] != self._non_key_value:
            raise ValueError("Never happen in set_non_key_value")

    @property
    def data(self):
        return self._data


def viterbi(model: Trie, tokens: str):
    def forward():
        length = len(tokens)
        strata = [{'solution':None, 'end':set(), 'min_loss':None, 'based_on':None} for i in range(length)]
        pending_start = None
        for start in range(length - 1): # the last one will be left behind
            # find edge from start to all already ends (end + 1)
            for end in range(start + 1, length):
                char = tokens[end - 1]
                if model.possible(char, start):
                    word = model.recent
                    if word:
                        strata[end]['end'].add(word)
                    if pending_start:
                        unk = tokens[pending_start:end - 1]
                        model.set_non_key_value(unk)
                        strata[end - 1]['end'].add(unk)
                        pending_start = None
                else:
                    if pending_start is None:
                        pending_start = start
                    break
            for s in strata:
                print(s, start)
            stratum = strata[start]
            if start == 0:
                stratum['min_loss'] = 0
            elif stratum['end']:
                path_loss_gen = ((tok, model[tok]+strata[start - len(tok)]['min_loss']) for tok in stratum['end'])
                tok, min_loss = min(path_loss_gen, key = lambda x:x[1])
                stratum['solution'] = tok
                stratum['based_on'] = start - len(tok)
                stratum['min_loss'] = min_loss
        return strata

    def backward(strata):
        based_on = -1
        solution = []
        while based_on:
            solution.append(strata[based_on]['solution'])
            based_on = strata[based_on]['based_on']
            return solution.reverse()
    return backward(forward())


def plot_2d_contour(x_y_entropy):
    X, Y, Z = x_y_entropy
    plt.figure()
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Grid Search')
    plt.show()
