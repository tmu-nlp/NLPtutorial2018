from pprint import pprint
from collections import defaultdict
import math


grammar_file = "../../test/08-grammar.txt"
input_file = "../../test/08-input.txt"


nonterms=[]
preterms=defaultdict(list)
for rule in open(grammar_file,"r",encoding="utf-8"):
	lhs, rhs, prob = rule.split("\t")
	rhs_symbols = rhs.split()
	if len(rhs_symbols) == 1:
		preterms[rhs].append((lhs,float(prob)))
	else:
		nonterms.append((lhs,rhs_symbols[0],rhs_symbols[1],float(prob)))


best_scores = defaultdict(lambda: -10000.0)
best_edges = {}
for line in open(input_file, "r", encoding="utf-8"):
    line = line.split()
    for idx, word in enumerate(line):
        l_idx = idx
        r_idx = idx + 1
        for lhs, prob in preterms[word]:
            best_scores[(lhs, l_idx, r_idx)] = -math.log(prob)
    pprint(best_scores)
    for j in range(2, len(line)+1):
        for i in reversed(range(j - 1)):
            for k in range(i + 1, j):
                for sym, lsym, rsym, logprob in nonterms:
                    logprob = -math.log(logprob)
                    if best_scores[(lsym, i, k)] > -10000.0 and best_scores[(
                            rsym, k, j)] > -10000.0:
                        my_lp = best_scores[(lsym, i, k)] + best_scores[(
                            rsym, k, j)] + logprob
                        if my_lp > best_scores[(sym, i, j)]:
                            best_scores[(sym, i, j)] = my_lp
                            best_edges[(sym, i, j)] = ((lsym, i, k), (rsym, k,
                                                                      j))

def print_tree(node):
    if node in best_edges:
        return "({node_tag} {best_node0} {best_node1})".format(
            node_tag=node[0],
            best_node0=print_tree(best_edges[node][0]),
            best_node1=print_tree(best_edges[node][1]))
    else:
        return "({node_tag} {word} )".format(
            node_tag=node[0],
            word=line[node[1]])

print(print_tree(("S",0,len(line))))