import dill
import copy
from tqdm import tqdm
from collections import defaultdict



def MakeFeats(stack,queue):
    feats = defaultdict(lambda: 0)
    
    if len(stack) > 0 and len(queue) > 0:
        w_0 = queue[0][1]
        w_1 = stack[-1][1]
        p_0 = queue[0][2]
        p_1 = stack[-1][2]
        feats[f'W-1{w_1},W-0{w_0}'] += 1
        feats[f'W-1{w_1},P-0{p_0}'] += 1
        feats[f'P-1{p_1},W-0{w_0}'] += 1
        feats[f'P-1{p_1},P-0{p_0}'] += 1
    if len(stack) > 1:
        w_1 = stack[-1][1]
        w_2 = stack[-2][1]
        p_1 = stack[-1][2]
        p_2 = stack[-2][2]
        feats[f'W-2{w_2},W-1{w_1}'] += 1
        feats[f'W-2{w_2},P-1{p_1}'] += 1
        feats[f'P-2{p_2},W-1{w_1}'] += 1
        feats[f'P-2{p_2},P-1{p_1}'] += 1

    return feats

def PredicScore(w,feats):
    s_r, s_l, s_s = 0, 0, 0
    for name, value in feats.items():
        s_r += w['right'][name] * value
        s_l += w['left'][name] * value
        s_s += w['shift'][name] * value
    return s_r, s_l, s_s

def calc_corr(stack, head, unproc):
    if stack[-1].head == stack[-2].id and stack[-1].unproc ==0:
        stack[-1].unproc -= 1
        corr ='right'
    elif stack[-2].head == stack[-1].id and stack[-2].unproc == 0:
        stack[-2].unproc -= 1
        corr = 'left'
    else:
        corr = 'shift'
    return corr


def calc_ans(s_s, s_r, s_l, queue):
    if s_s >= s_r and s_s >= s_l and len(queue) > 0:
        ans = 'shift'
    elif s_r >= s_l:
        ans = 'right'
    else:
        ans = 'left'
    
    return ans


def UpdateWeights(w,feats,ans,corr):
    for name, value in feats.items():
        w[ans][name] -= value
        w[corr][name] += value


        
        
def ShiftReduceTrain(queue,w,heads):
    heads = []
    stack = [(0,"ROOT","ROOT")]
    unproc = []
    for i in range(len(heads)):
        unproc.append(heads.count(i))
    while len(queue) >0 or len(stack) > 1:
        feats = MakeFeats(stack,queue)
        s_r, s_l, s_s = PredicScore(w,feats)
        corr = calc_corr(stack, head,unproc)
        ans = calc_ans(s_l,s_r,s_r,queue)
        if ans != corr:
            UpdateWeights(w,feats,ans,corr)
        #action according to corr
        if corr == 'shift':
            stack.append(queue.popleft())
        elif corr == 'left':
            unproc[stack[-1][0]] -= 1
            stack.remove(-2)
        elif corr == 'right':
            unproc[stack[-2][0]] -= 1
            stack.remove(-1)

if __name__ == '__main__':
    train_file = '../../data/mstparser-en-train.dep'
    epoch = 5
    data = list()
    queue = list()
    heads = [-1]
    with open(train_file)as f:
        for line in f:
            line = line.strip()
            if line == '':
                data.append(queue,heads)
                queue = list()
                heads = [-1]
            else:
                ID, word, base, pos, pos2, expand, head, label = line.split()
                queue.append((int(ID), word, pos))
                heads.append(int(head))
    w = {}
    w['right'] = defaultdict(lambda: 0)
    w['left'] = defaultdict(lambda: 0)
    w['shift'] = defaultdict(lambda: 0)
    for _ in tqdm(range(epoch)):
        for queue,heads in data:
            ShiftReduceTrain(queue.copy(),w,heads)
    with open('weight','w')as g:
        dill.dump(w,g)









