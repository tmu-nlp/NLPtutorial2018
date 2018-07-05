'''
../../data/mstparser-en-train.dep:
1	ms.	ms.	NNP	NNP	_	2	DEP
2	haag	haag	NNP	NNP	_	3	NP-SBJ
3	plays	plays	VBZ	VBZ	_	0	ROOT
4	elianti	elianti	NNP	NNP	_	3	NP-OBJ
5	.	.	.	.	_	3	DEP

            plays
    haag            elianti .
ms.

train:
python train_sr.py <../../data/mstparser-en-train.dep
'''
from collections import defaultdict, deque
import pickle as pkl
from sys import stdin

OP_SHIFT = 'SHIFT'
OP_LEFT = 'LEFT'
OP_RIGHT = 'RIGHT'
WEIGHT_PATH = 'weight.pkl'

def main():
    W = {}
    W[OP_SHIFT] = defaultdict(int)
    W[OP_LEFT] = defaultdict(int)
    W[OP_RIGHT] = defaultdict(int)

    data = load_data()

    # data = [list(data)[0]]

    c = 0
    for sentence in data:
        shift_reduce(deque(sentence[:]), W)
        c += 1

    print(f'{c} sentences learned.')

    with open(WEIGHT_PATH, 'wb') as f:
        pkl.dump(W, f)

def make_feats(stack, queue):
    feat_keys = []

    if len(queue) > 0 and len(stack) > 0:
        q = queue[0]
        s = stack[-1]
        feat_keys += bigram_feat_keys(s, q, 0)

    if len(stack) > 1:
        s1 = stack[-1]
        s2 = stack[-2]
        feat_keys += bigram_feat_keys(s2, s1, -1)
    
    feats = defaultdict(int)
    for key in feat_keys:
        feats[key] += 1
    
    return dict(feats)

def bigram_feat_keys(prev, next, i):
    return [
        f'W{i-1}{prev.word}|W{i}{next.word}',
        f'W{i-1}{prev.word}|P{i}{next.pos}',
        f'W{i-1}{prev.pos}|W{i}{next.word}',
        f'W{i-1}{prev.pos}|P{i}{next.pos}',
    ]
    
def shift_reduce(queue, W, mode='train'):
    heads = [None] * (len(queue)+1)
    stack = [Token(0, 'ROOT', 'ROOT', None, None)]

    while len(queue) > 0 or len(stack) > 1:
        # print(f'{stack} | {queue}')

        feats = make_feats(stack, queue)

        score = {}
        for key, weight in W.items():
            for feat, count in feats.items():
                score[key] = weight[feat] * count
        
        ans = predict_operation(score, len(queue))

        if mode == 'train':
            corr = get_gold_operation(stack, queue)
            for feat, count in feats.items():
                W[ans][feat] -= count
                W[corr][feat] += count
            ans = corr

        # do operation
        if corr == OP_SHIFT:
            stack.append(queue.popleft())
        elif corr == OP_LEFT:
            heads[stack[-2].id] = stack[-1].id
            del stack[-2]
        else:
            heads[stack[-1].id] = stack[-2].id
            del stack[-1]

    return heads

def predict_operation(scores, queue_len):
    ss, sl, sr = scores[OP_SHIFT], scores[OP_LEFT], scores[OP_RIGHT]

    if ss >= sl and ss >= sr and queue_len > 0:
        return OP_SHIFT
    elif sl >= sr:
        return OP_LEFT
    else:
        return OP_RIGHT

def get_gold_operation(stack, queue):
    if len(stack) < 2:
        return OP_SHIFT
    elif stack[-1].head == stack[-2].id and \
        all(map(lambda t: t.head != stack[-1].id, queue)):
        return OP_RIGHT
    elif stack[-2].head == stack[-1].id and \
        all(map(lambda t: t.head != stack[-2].id, stack[:-2])):
        return OP_LEFT
    else:
        return OP_SHIFT

class Token:
    def __init__(self, id, word, pos, head, attrs):
        self.id = id
        self.word = word
        self.pos = pos
        self.head = head
        self.attrs = attrs

    def __repr__(self):
        return self.word

def load_data(doc=stdin):
    sentence = []
    for line in stdin:
        line = line.rstrip().strip()
        if len(line) == 0:
            if len(sentence) > 0:
                yield sentence
                sentence = []
        else:
            attrs = line.split('\t')
            (rid, rname, rname1, rpos, rpos1, runder, rhead, rdep) = attrs
            token = Token(int(rid), rname, rpos, int(rhead), attrs)
            sentence.append(token)

if __name__ == '__main__':
    main()
    # import cProfile
    # cProfile.run('main()')