import math
from collections import defaultdict


# 木を出力
def s_exps(sym_i_j):
    sym, i, j = sym_i_j.split(' ')
    if sym_i_j in best_edge:  # 非終端記号
        return f'({sym} {s_exps(best_edge[sym_i_j][0])} {s_exps(best_edge[sym_i_j][1])})'
    else:  # 終端記号
        return f'({sym} {words[int(i)]})'

# 文法を読み込む
# grammer_path = '../../test/08-grammar.txt'  # テスト用
grammer_path = '../../data/wiki-en-test.grammar'  
nonterm = []  # （左、右１、右２、確率）の非終端記号
preterm = defaultdict(list)  # pre[右] = [（左、確率）...]形式のマップ
for line in open(grammer_path, 'r'):
    line = line.rstrip()
    lhs, rhs, prob = line.split('\t')
    rhs_symbols = rhs.split(' ')
    if len(rhs_symbols) == 1:  # 前終端記号
        preterm[rhs].append((lhs, math.log2(float(prob))))
    else:  # 非終端記号
        nonterm.append((lhs, rhs_symbols[0], rhs_symbols[1], math.log2(float(prob))))

# CKY
# input_path = '../../test/08-input.txt'  # テスト用
input_path = '../../data/wiki-en-short.tok'  
for line in open(input_path, 'r'):
    words = line.rstrip().split(' ')
    best_score = defaultdict(lambda: -math.inf)  # sym_ij : 最大対数確率
    best_edge = {}  # sym_ij : (lsym_ik, rsym_kj)

    # 前終端記号を追加
    for i in range(len(words)):
        for lhs, log_prob in preterm[words[i]]:
            best_score[f'{lhs} {i} {i+1}'] = log_prob

    # 非終端記号の組み合わせ
    for j in range(2, len(words)+1):  # jはスパンの右側
        for i in range(j-2, -1, -1):  # iはスパンの左側（右から左へ処理）
            for k in range(i+1, j):  # kはrsymの開始点
                # 各文法ルールを展開: log(P(sym → lsym rsym)) = logprob
                for sym, lsym, rsym, logprob in nonterm:
                    # 両方の子供の確率が0より大きい
                    if best_score[f'{lsym} {i} {k}'] > -math.inf and \
                       best_score[f'{rsym} {k} {j}'] > -math.inf:
                        # このノード・辺の対数確率を計算
                        my_lp = best_score[f'{lsym} {i} {k}'] + \
                                best_score[f'{rsym} {k} {j}'] + logprob
                        # この辺が確率最大のものなら更新
                        if my_lp > best_score[f'{sym} {i} {j}']:
                            best_score[f'{sym} {i} {j}'] = my_lp
                            best_edge[f'{sym} {i} {j}'] = (f'{lsym} {i} {k}', f'{rsym} {k} {j}')

    # 構文木の出力
    print(s_exps(f'S 0 {len(words)}'))