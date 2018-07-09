import sys
sys.path.append('..')
from utils.data import load_text
from collections import defaultdict
from math import log2

def load_grammar(fname = '../../test/08-grammar.txt'):
    terms = defaultdict(list)
    nonterms = []
    for parent, children, prob in load_text(fname, '\t'):
        children = tuple(children.split(' '))
        entropy = log2(float(prob))

        if len(children) == 1:
            terms[children[0]].append((parent, entropy))
        else:
            nonterms.append((parent, entropy) + children )
    return terms, nonterms

def unk(terms):
    unk_entropy = defaultdict(list)
    sum_entropy = 0
    for child, possible_parents in terms.items():
        for parent, entropy in possible_parents:
            unk_entropy[parent].append(entropy)
            sum_entropy += entropy

    # give unk an average entropy over all syms
    for parent, entropys in unk_entropy.items():
        # list of entropy -> entropy
        unk_entropy[rhs] = sum(entropys) / sum_entropy
    *unk_entropy, = zip(unk_entropy.keys(), unk_entropy.values())

    # for unk (shared_parents, average entropy)
    terms.default_factory = (lambda: unk_entropy)

def bottom_up_ij(length):
    coverage = 1
    while coverage <= length:
        step = coverage - 1
        while step >= 0:
            yield step, coverage + 1 # plus 1 for slicing & ranging
            step -= 1
        coverage += 1

def bottom_up(tokens, grammar):

    terms, nonterms = grammar
    entropy_bag = {}
    local_best_coverage = {}

    # terms bottom
    for i, tok in enumerate(tokens):
        # 'a' can be [noun letter 'a', det, ...]
        for child, entropy in terms[tok]:
            entropy_bag[(child, i, i + 1)] = entropy

    # nonterms up
    for i, j in bottom_up_ij(len(tokens)):
        for k in range(i+1, j):
            # try every grammar!
            for parent, entropy, lchild, rchild in nonterms:
                    ij = (parent, i, j)
                    ik = (lchild, i, k)
                    kj = (rchild, k, j)
                    if (ik in entropy_bag) and (kj in entropy_bag):
                        ij_entropy = entropy_bag[ik] + entropy_bag[kj] + entropy
                        if (ij not in entropy_bag) or (ij_entropy < entropy_bag[ij]):
                            entropy_bag[ij] = ij_entropy
                            local_best_coverage[ij] = (ik, kj)
                            final_coverage = ij
    return local_best_coverage, final_coverage

def top_down(local_best_coverage, tokens, ij, level = 0):

    postag, i, _ = ij
    indent = ' ' * level
    if ij in local_best_coverage:
        # nonterms
        ik, kj = local_best_coverage[ij]
        ik_str = top_down(local_best_coverage, tokens, ik, level + 1)
        kj_str = top_down(local_best_coverage, tokens, kj, level + 1)
        return '%s%s (\n%s \n%s\n%s)' % (indent, postag, ik_str, kj_str, indent)
    else:
        # terms
        return "%s[%s '%s']" % (indent, postag, tokens[i])


if __name__ == '__main__':
    grammar = load_grammar('../../test/08-grammar.txt')
    for tokens in load_text('../../test/08-input.txt'):
        coverage_bag, root = bottom_up(tokens, grammar)
        print(top_down(coverage_bag, tokens, root))
