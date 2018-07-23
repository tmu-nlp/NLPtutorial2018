from collections import deque
from collections import defaultdict
import dill
import copy
from tqdm import tqdm

train_path = '../../data/mstparser-en-train.dep'
epoch = 5


def make_features(stack, queue):
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


def calc_score(feats, w):
    s_r, s_l, s_s = 0, 0, 0
    for name, value in feats.items():
        s_r += w['right'][name] * value
        s_l += w['left'][name] * value
        s_s += w['shift'][name] * value
    
    return s_r, s_l, s_s


def calc_correct(stack, heads, unproc):
    if len(stack) < 2:
        correct = 'shift'
    # 右が左の親で左に未処理の子供がいない
    elif heads[stack[-1][0]] == stack[-2][0] and unproc[stack[-1][0]] == 0: 
        correct = 'right'
    # 左が右の親で右に未処理の子供がいない
    elif heads[stack[-2][0]] == stack[-1][0] and unproc[stack[-2][0]] == 0:
        correct = 'left'
    else:
        correct = 'shift'
    
    return correct


def calc_predict(s_r, s_l, s_s, queue):
    if s_s >= s_r and s_s >= s_l and len(queue) > 0:
        predict = 'shift'
    elif s_r >= s_l:
        predict = 'right'
    else:
        predict = 'left'
    
    return predict


def update_weights(w, predict, correct, feats):
    for name, value in feats.items():
        w[predict][name] -= value
        w[correct][name] += value


def shift_reduce_train(queue, heads, weights):
    stack = [(0, 'ROOT', 'ROOT')]
    unproc = []  # 各単語の未処理の子供の数

    # unprocの初期値を計算
    for i in range(len(heads)):
        unproc.append(heads.count(i))

    while len(queue) > 0 or len(stack) > 1:
        feats = make_features(stack, queue)

        # shift, right, leftのスコアと正解を計算
        s_r, s_l, s_s = calc_score(feats, weights)
        correct = calc_correct(stack, heads, unproc)

        # 予測を計算
        predict = calc_predict(s_r, s_l, s_s, queue)

        # 重み更新
        if predict != correct:
            update_weights(weights, predict, correct, feats)

        # correctに基づいて実行
        if correct == 'shift':
            stack.append(queue.popleft())  # shiftを実行
        elif correct == 'left':  # reduce leftを実行
            unproc[stack[-1][0]] -= 1
            del stack[-2]
        elif correct == 'right':  # reduce rightを実行
            unproc[stack[-2][0]] -= 1
            del stack[-1]

if __name__ == '__main__':   

    # 重み
    weights = {}
    weights['right'] = defaultdict(lambda: 0)
    weights['left'] = defaultdict(lambda: 0)
    weights['shift'] = defaultdict(lambda: 0)

    data = []
    queue = deque()
    heads = [-1]

    # 文ごとのqueueとheadsをdataに追加
    for line in open(train_path, 'r'):
        if line == '\n':
            data.append((queue, heads))
            queue = deque()
            heads = [-1]
        else:
            id_, word, _, pos, _, _, parent, _ = line.rstrip('\n').split('\t')
            queue.append((int(id_), word, pos))
            heads.append(int(parent))

    # 学習
    for _ in tqdm(range(epoch)):
        for queue, heads in data:
            shift_reduce_train(queue.copy(), heads, weights)
    
    # ファイル書き出し
    with open('weights', 'wb') as f:
        dill.dump(weights, f)