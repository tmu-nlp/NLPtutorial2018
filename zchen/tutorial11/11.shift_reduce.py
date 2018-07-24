import sys
sys.path.append('..')
from utils.data import CoNLL_gen, Token
from collections import defaultdict
from tqdm import tqdm

op_shift = 0
op_reduce_left = 1
op_reduce_right = 2

def golden_op(stack, queue):
    if len(queue) == 0: # when source is empty: 2 ops
        if stack[-1].head == stack[-2].id:
            return op_reduce_right
        else:
            return op_reduce_left

    # source is not empty: 3 ops
    # shift all the dependency by in_degree, prevent greedy on right
    if stack[-1].head == stack[-2].id and not stack[-1].in_degree:
        return op_reduce_right
    elif stack[-2].head == stack[-1].id and not stack[-2].in_degree:
        return op_reduce_left
    else:
        return op_shift

def make_feats(stack, queue):
    phi = {}
    s1 = stack[-1]
    if queue and stack:
        q = queue[0]
        phi.update({
            f'W-1 {s1.word} W-0 {q.word}': 1,
            f'W-1 {s1.word} P-0 {q.pos}': 1,
            f'P-1 {s1.pos}  W-0 {q.word}': 1,
            f'P-1 {s1.pos}  P-0 {q.pos}': 1, })
        # no effect
        #if len(stack) > 1:
        #    s2 = stack[-2]
        #    phi.update({
        #        f'W-2 {s2.word} W-0 {q.word}': 1,
        #        f'W-2 {s2.word} P-0 {q.pos}': 1,
        #        f'P-2 {s2.pos}  W-0 {q.word}': 1,
        #        f'P-2 {s2.pos}  P-0 {q.pos}': 1, })
    if len(stack) > 1:
        s2 = stack[-2]
        phi.update({
            f'W-2 {s2.word} W-1 {s1.word}': 1,
            f'W-2 {s2.word} P-1 {s1.pos}': 1,
            f'P-2 {s2.pos} W-1 {s1.word}': 1,
            f'P-2 {s2.pos} P-1 {s1.pos}': 1, })
    return phi

def shift_reduce(op_ws, queue, train_mode):
    heads = [0] * len(queue)
    stack = [Token(0, 'ROOT', 'ROOT')]
    while queue or len(stack) > 1:
        if len(stack) <= 1:
            stack.append(queue.popleft())
            continue

        feats = make_feats(stack, queue)
        op = project(op_ws, feats, queue)
        if train_mode:
            gold = golden_op(stack, queue)
            update(op_ws, feats, op, corr)
            op = gold

        if op == op_shift:
            stack.append(queue.popleft())
        elif op == op_reduce_left:
            if train_mode:
                stack[-1].in_degree -= 1
            else:
                heads[stack[-2].id - 1] = stack[-1].id
            stack.pop(-2)
        elif op == op_reduce_right:
            if train_mode:
                stack[-2].in_degree -= 1
            else:
                heads[stack[-1].id - 1] = stack[-2].id
            stack.pop(-1)
        else:
            raise Exception("???")
    if not train_mode:
        return heads

def update(op_ws, feats, op, golden_op):
    if op == golden_op:
        return
    for feat, value in feats.items():
        op_ws[op][feat] -= value
        op_ws[golden_op][feat] += value

def project(op_ws, feats, queue):
        ss, sl, sr = 0, 0, 0
        for feat, value in feats.items():
            ss += op_ws[op_shift][feat] * value
            sl += op_ws[op_reduce_left][feat] * value
            sr += op_ws[op_reduce_right][feat] * value

        if queue and ss >= sl and ss >= sr:
            return op_shift
        elif sl >= sr:
            return op_reduce_left
        else:
            return op_reduce_right

class ShiftReduceDependency:
    def __init__(self):
        self._op_ws = {op: defaultdict(int) for op in range(op_reduce_right + 1)}

    def train(self, fname, num_epochs):
        for _ in tqdm(range(num_epochs), desc = 'Training'):
            for queue in CoNLL_gen(fname):
                shift_reduce(self._op_ws, queue, train_mode = True)

    def test(self, fname):
        out_file = 'test.dep'
        print('Test:', out_file)
        with open(out_file, 'w') as fw:
            for queue in CoNLL_gen(fname):
                heads = shift_reduce(self._op_ws, queue, train_mode = False)
                result = '\n'.join(('_\t' * 6 + '%d\t_\t') % i for i in heads)
                fw.writelines(result)
                fw.write('\n\n')


if __name__ == '__main__':
    sr = ShiftReduceDependency()
    sr.train('../../data/mstparser-en-train.dep', 5)
    sr.test('../../data/mstparser-en-test.dep')
