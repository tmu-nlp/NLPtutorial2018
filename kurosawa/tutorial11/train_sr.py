from collections import defaultdict
import pickle
import random
import copy

def MakeFeatures(stack, queue):
    features = defaultdict(int)
    if len(stack) >0 and len(queue) >0:
        features["W-1"+stack[-1][1]+",W0"+queue[0][1]] += 1
        features["W-1"+stack[-1][1]+",P0"+queue[0][2]] += 1
        features["P-1"+stack[-1][2]+",W0"+queue[0][1]] += 1
        features["P-1"+stack[-1][2]+",P0"+queue[0][2]] += 1
    if len(stack) > 1:
        features["W-2"+stack[-2][1]+",W-1"+stack[-1][1]] += 1
        features["W-2"+stack[-2][1]+",P-1"+stack[-1][2]] += 1
        features["P-2"+stack[-2][2]+",W-1"+stack[-1][1]] += 1
        features["P-2"+stack[-2][2]+",P-1"+stack[-1][2]] += 1
    return features

def PredictScore(W,features):
    score = [0,0,0]
    for key,value in features.items():
        for i in range(3):
            score[i] += W[i][key] * value
    return score

def UpdateWeights(W, features, predict, correct):
    for key,value in features.items():
        if predict == "shift":
            W[0][key] -= value
        elif predict == "left":
            W[1][key] -= value
        else:
            W[2][key] -= value
        if correct == "shift":
            W[0][key] += value
        elif correct == "left":
            W[1][key] += value
        else:
            W[2][key] += value

def ShiftReduceTrain(queue, heads, W):
    stack = [(0, "ROOT", "ROOT")]
    unproc = list()
    for i in range(len(heads)):
        unproc.append(heads.count(i))
    while len(queue) > 0 or len(stack) > 1:
        features = MakeFeatures(stack, queue)
        score = PredictScore(W, features)
        if (score[0] > score[1] and score[0] > score[2] and len(queue) > 0) or len(stack) < 2:
            predict = "shift"
        elif score[1] > score[0] and score[1] > score[2]:
            predict = "left"
        else:
            predict = "right"
        if len(stack) < 2:
            correct = "shift"
            stack.append(queue.pop(0))
        elif heads[stack[-1][0]] == stack[-2][0] and unproc[stack[-1][0]] == 0:
            correct = "right"
            unproc[stack[-2][0]] -= 1
            stack.pop(-1)
        elif heads[stack[-2][0]] == stack[-1][0] and unproc[stack[-2][0]] == 0:
            correct = "left"
            unproc[stack[-1][0]] -= 1
            stack.pop(-2)
        else:
            correct = "shift"
            stack.append(queue.pop(0))
        if predict != correct:
            UpdateWeights(W, features, predict, correct)

if __name__ == '__main__':
    train_file = '../../data/mstparser-en-train.dep'
    weight_file = 'weight.pickle'
    epoch = 15
    data = list()
    queue = list()
    heads = [-1]
    with open(train_file) as train_data:
        for line in train_data:
            line = line.strip()
            if line == '':
                data.append((queue,heads))
                queue = list()
                heads = [-1]
            else:
                ID, word, base, pos, pos2, expand, head, label = line.split()
                ID = int(ID)
                head = int(head)
                queue.append((ID, word, pos))
                heads.append(head)
    W_shift = defaultdict(int)
    W_left = defaultdict(int)
    W_right = defaultdict(int)
    W = [W_shift, W_left, W_right]
    for epoch_now in range(epoch):
        random.shuffle(data)
        data_ = copy.deepcopy(data)
        for queue, heads in data_:
            ShiftReduceTrain(queue, heads, W)
    with open(weight_file, 'wb') as weight_data:
        pickle.dump(W, weight_data)
