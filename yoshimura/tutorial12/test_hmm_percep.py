import numpy as np
import dill
from collections import defaultdict
from train_hmm_percep import hmm_viterbi

# test_path = '../../test/05-test-input.txt'  # テスト用パス
test_path = '../../data/wiki-en-test.norm'

if __name__ == '__main__':
    # idsと重みとpossible_tagsの読み込み
    with open('ids', 'rb') as f:
        ids = dill.load(f)
    with open('weights', 'rb') as f:
        weights = dill.load(f)
    with open('p_tags', 'rb') as f:
        possible_tags = dill.load(f)

    # テスト
    with open('result', 'w') as f:
        for line in open(test_path, 'r'):
            X = line.rstrip('\n').split(' ')
            Y_hat = hmm_viterbi(weights, X, ids, possible_tags)
            line = ' '.join(Y_hat) + '\n'
            f.write(f'{line}')
    
# ../../script/gradepos.pl ../../data/wiki-en-test.pos result 
# Accuracy: 86.13% (3930/4563) epoch5
