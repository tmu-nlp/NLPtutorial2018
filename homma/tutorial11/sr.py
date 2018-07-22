import math
import os
# import queue

from itertools import product
from collections import defaultdict
from collections import namedtuple
from collections import deque
from tqdm import tqdm as tq


def make_feats(stack, queue):
    phi = {}
    s1 = stack[-1]
    if queue and stack:
        q = queue[0]
        phi.update({
            f'W-1 {s1.word} W-0 {q.word}': 1,
            f'W-1 {s1.word} P-0 {q.pos}': 1,
            f'P-1 {s1.pos} W-0 {q.word}': 1,
            f'P-1 {s1.pos} P-0 {q.pos}': 1, })
    if len(stack) > 1:
        s2 = stack[-2]
        phi.update({
            f'W-2 {s2.word} W-1 {s1.word}': 1,
            f'W-2 {s2.word} P-1 {s1.pos}': 1,
            f'P-2 {s2.pos} W-1 {s1.word}': 1,
            f'P-2 {s2.pos} P-1 {s1.pos}': 1,})
    return phi


def calc_corr(stack, queue):
    if not queue:
        # 強制REDUCE
        if stack[-1].head == stack[-2].id_:
            return 'RIGHT'
        else:
            return 'LEFT'

    if stack[-1].head == stack[-2].id_ and not stack[-1].unproc[0]:
        return 'RIGHT'
    elif stack[-2].head == stack[-1].id_ and not stack[-2].unproc[0]:
        return 'LEFT'
    else:
        return 'SHIFT'


def calc_ans(feats, queue):
    s_s = s_l = s_r = 0
    for key, value in feats.items():
        s_s += w['SHIFT'][key] * value
        s_l += w['LEFT'][key] * value
        s_r += w['RIGHT'][key] * value

    if s_s >= s_l and s_s >= s_r and queue:
        return 'SHIFT'
    elif s_l >= s_r:
        return 'LEFT'
    else:
        return 'RIGHT'


def update(feats, w, ans, corr):
    if ans == corr:
        return
    for key, value in feats.items():
        w[ans][key] -= value
        w[corr][key] += value


def shift_reduce(queue):
    result = ['_\t'*6+'0\t_\n'] * len(queue)
    stack = [Token(0, 'ROOT', 'ROOT', None, [0], -1)]
    while queue or len(stack) > 1:
        if len(stack) <= 1:
            # 強制SHIFT
            stack.append(queue.popleft())
            continue

        feats = make_feats(stack, queue)
        ans = calc_ans(feats, queue)

        # SHIFT-REDUCE実行
        if ans == 'SHIFT':
            stack.append(queue.popleft())
        elif ans == 'LEFT':
            # stack[-2].pred = stack[-1].id_
            result[stack[-2].id_ - 1] = '_\t' * 6 + f'{stack[-1].id_}\t_\n'
            del stack[-2]
        else:
            # stack[-1].pred = stack[-2].id_
            result[stack[-1].id_ - 1] = '_\t' * 6 + f'{stack[-2].id_}\t_\n'
            del stack[-1]

    return result


def shift_reduce_train(queue):
    stack = [Token(0, 'ROOT', 'ROOT', None, [0], -1)]
    while queue or len(stack) > 1:
        if len(stack) <= 1:
            # 強制SHIFT
            stack.append(queue.popleft())
            continue

        feats = make_feats(stack, queue)
        ans = calc_ans(feats, queue)
        corr = calc_corr(stack, queue)
        update(feats, w, ans, corr)

        # SHIFT-REDUCE実行
        if corr == 'SHIFT':
            stack.append(queue.popleft())
        elif corr == 'LEFT':
            # stack[-2].pred = stack[-1].id_
            stack[-1].unproc[0] -= 1
            del stack[-2]
        else:
            # stack[-1].pred = stack[-2].id_
            stack[-2].unproc[0] -= 1
            del stack[-1]


def load_mst(path):
    # 1文が終わる毎に空行が入っている
    sentence = []
    for raw_line in open(path, encoding='utf8'):
        line = raw_line.rstrip()
        if line:
            id_, word, _, pos, _, _, head, _ = line.split('\t')
            token = Token(int(id_), word, pos, int(head), [0], -1)
            sentence.append(token)
        elif sentence:
            yield sentence
            sentence = []


def init_unproc(queue):
    for token in queue:
        if token.head:
            queue[token.head - 1].unproc[0] += 1


def train(path):
    for epoch in range(1):
    # for epoch in tq(range(10)):
        for sentence in load_mst(path):
            queue = deque(sentence)
            init_unproc(queue)
            shift_reduce_train(queue)


def test(path):
    with open('ans.dep', 'w', encoding='utf8') as f:
        for sentence in load_mst(path):
            queue = deque(sentence)
            result = shift_reduce(queue)
            f.writelines(result)
            f.write('\n')


def main():
    train_path = '../../data/mstparser-en-train.dep'.replace('/', os.sep)
    test_path = '../../data/mstparser-en-test.dep'.replace('/', os.sep)
    train(train_path)
    test(test_path)


if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    w = {
        'SHIFT': defaultdict(float),
        'LEFT': defaultdict(float),
        'RIGHT': defaultdict(float)}
    Token = namedtuple('Token', ['id_', 'word', 'pos', 'head', 'unproc', 'pred'])
    main()


'''
$ ../../script/grade-dep.py ans.dep ../../data/mstparser-en-test.dep
61.543436% (2855/4639)
'''
