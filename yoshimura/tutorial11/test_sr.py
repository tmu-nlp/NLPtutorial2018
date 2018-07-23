import dill
from collections import deque
from collections import defaultdict
from train_sr import make_features, calc_score

test_path = '../../data/mstparser-en-test.dep'


def shift_reduce(queue, weights):
    heads = [-1] * (len(queue) + 1)
    stack = [(0, 'ROOT', 'ROOT')]

    while len(queue) > 0 or len(stack) > 1:
        feats = make_features(stack, queue)

        # shift, right, leftのスコアと正解を計算
        s_r, s_l, s_s = calc_score(feats, weights)

        if len(stack) < 2 or (s_s >= s_l and s_s >= s_r and len(queue) > 0):  # shiftを実行
            stack.append(queue.popleft())
        elif s_l >= s_r:  # reduce leftを実行
            heads[stack[-2][0]] = stack[-1][0]
            del stack[-2]
        else:  # reduce rightを実行
            heads[stack[-1][0]] = stack[-2][0]
            del stack[-1]
    
    return heads

if __name__ == '__main__':
    # 重み読み込み
    with open('weights', 'rb') as f:
        weights = dill.load(f)

    data = []
    queue = deque()
    conll = []
    conll_list = []

    # 文ごとのqueueとconllを作成
    for line in open(test_path, 'r'):
        if line == '\n':
            data.append(queue)
            conll_list.append(conll)
            queue = deque()
            conll = []
        else:
            id_, word, ori, pos1, pos2, ext, parent, label = line.strip('\n').split('\t')
            queue.append((int(id_), word, pos1))
            conll.append([id_, word, ori, pos1, pos2, ext, parent, label])

    # テスト
    heads_list = []
    for queue in data:
        heads_list.append(shift_reduce(queue, weights))
    
    # 結果出力
    with open('result', 'w') as f:
        for conll, heads in zip(conll_list, heads_list):
            for types, head in zip(conll, heads[1:]):
                types[6] = str(head)
                line = '\t'.join(types) + '\n'
                f.write(f'{line}')
            f.write('\n')

# ../../script/grade-dep.py ../../data/mstparser-en-test.dep result
# 65.596034% (3043/4639)