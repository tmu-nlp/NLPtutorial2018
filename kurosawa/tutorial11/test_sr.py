import pickle
from train_sr import *

def ShiftReduce(queue, W):
    stack = [(0, "ROOT", "ROOT")]
    heads = [-1] * (len(queue) + 1)
    while len(queue) > 0 or len(stack) > 1:
        features = MakeFeatures(stack, queue)
        score = PredictScore(W, features)
        if (score[0] > score[1] and score[0] > score[2] and len(queue) > 0) or len(stack) < 2:
            stack.append(queue.pop(0))
        elif score[1] > score[0] and score[1] > score[2]:
            heads[stack[-2][0]] = stack[-1][0]
            stack.pop(-2)
        else:
            heads[stack[-1][0]] = stack[-2][0]
            stack.pop(-1)
    return heads

if __name__ == '__main__':
    test_file = '../../data/mstparser-en-test.dep'
    weight_file = 'weight.pickle'
    answer_file = 'answer.dep'
    data = list()
    queue = list()
    lines = list()
    liness = list()
    with open(test_file) as test_data:
        for line in test_data:
            line = line.strip()
            if line == '':
                data.append(queue)
                queue = list()
                liness.append(lines)
                lines = list()
            else:
                ID, word, base, pos, pos2, expand, head, label = line.split()
                ID = int(ID)
                head = int(head)
                queue.append((ID, word, pos))
                lines.append(line.split())
    with open(weight_file, 'rb') as weight_data:
        W = pickle.load(weight_data)
    with open(answer_file, 'w') as answer_data:
        for i, queue in enumerate(data):
            heads = ShiftReduce(queue, W)
            heads.pop(0)
            for j in range(len(heads)):
                liness[i][j][6] = str(heads[j])
                answer_data.write('{}\n'.format('\t'.join(liness[i][j])))
            answer_data.write('\n')
