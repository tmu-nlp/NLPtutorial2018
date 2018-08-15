import pickle
from train import *

def ShiftReduce(queue, w):
    heads = [-1] * (len(queue) + 1)
    stack = [(0, 'ROOT', 'ROOT')]

    while len(queue) > 0 or len(stack) > 1:
        feats = MakeFeats(stack, queue)

        # shift, right, leftのスコアと正解を計算
        s_r, s_l, s_s = PredicScore(feats, w)

        if len(stack) < 2 or (s_s >= s_l and s_s >= s_r and len(queue) > 0): 
            stack.append(queue.popleft())
        elif s_l >= s_r:  
            heads[stack[-2][0]] = stack[-1][0]
            stack.pop(-2)
        else: 
            heads[stack[-1][0]] = stack[-2][0]
            stack.pop(-1)
    return heads

if __name__ == '__main__':
    test_file = '../../data/mstparser-en-test.dep'
    
    data = list()
    queue = list()
    conll = list()
    conll_list = list()
    with open(test_file) as test_data:
        for line in test_data:
            line = line.strip()
            if line == '':
                data.append(queue)
                queue = list()
                conll_list.append(conll)
                conll = list()
            else:
                ID, word, base, pos, pos2, expand, head, label = line.split()
                ID = int(ID)
                head = int(head)
                queue.append((ID, word, pos))
                conll.append(line.split())
    with open('weight', 'rb') as g:
        w = pickle.load(g)
    with open('answer', 'w') as answer_data:
        for i, queue in enumerate(data):
            heads = ShiftReduce(queue, w)
            heads.pop(0)
            for j in range(len(heads)):
                conll_list[i][j][6] = str(heads[j])
                answer_data.write('{}\n'.format('\t'.join(conll_list[i][j])))
            answer_data.write('\n')
    