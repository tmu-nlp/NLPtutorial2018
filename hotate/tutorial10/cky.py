# -*- coding: utf-8 -*-
from term import Preterm, Nonterm, Best


def grammar(preterm, nonterm, gram_file):
    for rule in open(gram_file, 'r'):
        lhs, rhs, prob = rule.strip().split('\t')
        rhs = rhs.split(' ')
        if len(rhs) == 1:
            preterm(lhs, rhs[0], prob)
        else:
            nonterm(lhs, rhs[0], rhs[1], prob)


def add_preterm(preterm, words):
    best = Best()
    for i, word in enumerate(words):
        for sym, prob in preterm.index(word):
            best.scores(i, i+1, sym, prob)
    return best


def forward(nonterm, best, words):
    for r in range(2, len(words)+1):
        for l in range(r-2, -1, -1):
            for m in range(l+1, r):
                nonterm.update(best, r, l, m)


def print_(sym_name, best, words):
    sym = sym_name.split()
    if sym_name in best.edge:
        return f'({sym[0]} {print_(best.edge[sym_name][0], best, words)} {print_(best.edge[sym_name][1], best, words)})'
    else:
        return f'({sym[0]} {words[int(sym[1])]})'


def main(gram_file, sent_file):
    preterm = Preterm()
    nonterm = Nonterm()
    grammar(preterm, nonterm, gram_file)
    with open('out.txt', 'w') as f:
        for sentence in open(sent_file, 'r'):
            words = sentence.strip().split()
            best = add_preterm(preterm, words)
            forward(nonterm, best, words)
            sym_name = f'S 0 {len(words)}'
            print(print_(sym_name, best, words), file=f)


if __name__ == '__main__':
    # main('../../test/08-grammar.txt', '../../test/08-input.txt')
    main('../../data/wiki-en-test.grammar', '../../data/wiki-en-short.tok')
