from collections import defaultdict
import dill

def train(data_train):
    context, transition, emit, Prob_T, Prob_E = [defaultdict(int) for i in range(5)]
    for line in data_train:
        previous = '<s>'
        context[previous] += 1
        wordtags = line.split()
        for wordtag in wordtags:
            word, tag = wordtag.split('_')
            transition[previous + ' ' + tag] += 1
            context[tag] += 1
            emit[tag + ' ' + word] += 1
            previous = tag
        transition[previous + ' </s>'] += 1
    for key, value in transition.items():
        previous, word = key.split()
        Prob_T[key] = value/context[previous]
    for key, value in emit.items():
        tag, word = key.split()
        Prob_E[key] = value/context[tag]
    return Prob_T, Prob_E, context

if __name__ == '__main__':
    with open('../../data/wiki-en-train.norm_pos', 'r') as data_in:
        Prob_T, Prob_E, context = train(data_in)
    with open('result/result_train.dump', 'wb') as data_out:
        dill.dump([Prob_T, Prob_E, context], data_out)