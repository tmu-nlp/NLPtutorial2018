from .n_gram import N_Gram_Family, log_gen
from collections import defaultdict
import numpy as np
from tqdm import tqdm
try:
    import matplotlib.pyplot as plt
except RuntimeError:
    import sys
    sys.stderr.write("utils.solution module: Plot function with matplotlib will be unavailable.\n")
_log = log_gen()


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


def viterbi(model, tokens, max_len):
    def forward():
        cache = {}
        def possible_at(start):
            for l in range(max_len):
                end = start+l+1
                tok = tokens[start:end]
                if tok in model.raw_count:
                    cache[tok] = -_log(model.prob_of(tok))
                    yield start, end, tok

        length = len(tokens)
        strata = [{'solution':None, 'end':set(), 'min_loss':None, 'based_on':None} for i in range(length)]
        for pos in range(length):
            for start, end, tok in possible_at(pos):
                strata[end]['end'].add(tok)
            path_loss_gen = (tok, cache[tok]+strata[pos-len(tok)]['min_loss'] for tok in strata[pos]['end'])
            tok, min_loss = min(path_loss_gen, key x:x[1])
            strata[pos]['solution'] = tok
            strata[pos]['based_on'] = pos - len(tok)
            strata[pos]['min_loss'] = min_loss
        return strata

    def backward(strata):
        based_on = -1
        solution = []
        while based_on:
            solution.append(strata[based_on]['solution'])
            based_on = strata[based_on]['based_on']
            return solution[::-1]
    return backward(forward())


def plot_2d_contour(x_y_entropy):
    X, Y, Z = x_y_entropy
    plt.figure()
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Simplest default with labels')
