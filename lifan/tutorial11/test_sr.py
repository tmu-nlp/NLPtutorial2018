from train_sr import *
import dill

def ShiftReduce(queue, weights):
    stack = [(0, 'ROOT', 'ROOT')]
    heads = [-1 for i in range(len(queue) + 1)]

    while len(queue) > 0 or len(stack) > 1:
        features = MakeFeatures(stack, queue)
        scores = {}
        scores['SHIFT'] = PredictScore(weights['SHIFT'], features)
        scores['LEFT'] = PredictScore(weights['LEFT'], features)
        scores['RIGHT'] = PredictScore(weights['RIGHT'], features)

        if (max(scores.items(), key=lambda x: x[1])[0] == 'SHIFT' and len(queue) > 0) or len(stack) < 2:
            stack.append(queue.pop(0))
        elif max(scores.items(), key=lambda x: x[1])[0] == 'LEFT':
            heads[stack[-2][0]] = stack[-1][0]
            stack.pop(-2)
        else:
            heads[stack[-1][0]] = stack[-2][0]
            stack.pop(-1)

    return heads

if __name__ == '__main__':
    path_data_test = '../../data/mstparser-en-test.dep'
    path_data_out = 'my_answer.txt'
    path_weights = 'weights.dump'

    data = MakeData(path_data_test)
    with open(path_weights, 'rb') as data_weights:
        weights = dill.load(data_weights)

    with open(path_data_test) as data_in, open(path_data_out, 'w') as data_out:
        for queue in map(lambda x: x[0], data):
            heads = ShiftReduce(queue, weights)
            for i, line in enumerate(data_in):
                if line == '\n':
                    print(file=data_out)
                    break
                else:
                    print('\t'.join(line.strip().split('\t')[0:6] + [str(heads[i+1])] + [line.strip().split('\t')[-1]]), file=data_out)