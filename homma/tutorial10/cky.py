import math
import nltk
import os

from itertools import product
from collections import defaultdict


def ijk_iter(words, nonterm):
    'lsym_ik, rsym_kj, sym_ij, logprob'
    for j in range(2, len(words) + 1): # スパンの右側
        for i in range(j - 2, -1, -1): # スパンの左側（右から左へ）
            for k in range(i + 1, j):  # rsym の開始点
                for sym, lsym, rsym, logprob in nonterm:
                    yield f'{lsym} {i} {k}', f'{rsym} {k} {j}', f'{sym} {i} {j}', logprob


def main():
    grammar_path = '../../test/08-grammar.txt'.replace('/', os.sep)
    input_path = '../../test/08-input.txt'.replace('/', os.sep)
    # grammar_path = '../../data/wiki-en-test.grammar'.replace('/', os.sep)
    # input_path = '../../data/wiki-en-short.tok'.replace('/', os.sep)

    MINUS_INF = -1e100

    # "lhs \\t rhs \\t prob \\n" 形式の文法を読み込む
    nonterm = [] # （左，右１，右２，確率）の非終端記号
    preterm = defaultdict(list) # pre[右] = [(左, 確率)...] 形式の辞書
    for rule in open(grammar_path, encoding='utf8'):
        lhs, rhs, prob_str = rule.rstrip().split('\t')
        log_prob = math.log(float(prob_str))
        rhss = rhs.split()
        if len(rhss) == 1:
            preterm[rhs].append((lhs, log_prob))
        else:
            nonterm.append((lhs, rhss[0], rhss[1], log_prob))

    for line in open(input_path, encoding='utf8'):
        words = line.strip().split()
        best_score = defaultdict(lambda: MINUS_INF) # [sym_i,j] = 最大対数確率
        best_edge = {} # [sym_i,j] = (lsym_i,k , rsym_k,j)

        # 前終端記号を追加
        for i in range(len(words)):
            for lhs, log_prob in preterm[words[i]]:
                best_score[f'{lhs} {i} {i+1}'] = log_prob

        # 非終端記号の組み合わせ
        for lsym_ik, rsym_kj, sym_ij, logprob in ijk_iter(words, nonterm):
            if lsym_ik in best_score and rsym_kj in best_score:
                my_lp = best_score[lsym_ik] + best_score[rsym_kj] + logprob
                if my_lp > best_score[sym_ij]:
                    best_score[sym_ij] = my_lp
                    best_edge[sym_ij] = (lsym_ik, rsym_kj)

        def tree(sym_ij):
            '木をS式で返す'
            sym, i_, _ = sym_ij.split()
            i = int(i_)
            if sym_ij in best_edge:
                return f'({sym} {tree(best_edge[sym_ij][0])} {tree(best_edge[sym_ij][1])})'
            else: # 終端記号
                return f'({sym} {words[i]})'

        # 木を出力
        print(tree(f'S 0 {len(words)}'))


if __name__ == '__main__':
    # os.chdir(os.path.dirname(__file__))
    main()
