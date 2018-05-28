# python test_perceptron.py > ans.labeled
# ../../script/grade-prediction.py ../../data/titles-en-test.labeled ans.labeled

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.pardir)

    from common.binary_classifier import BinaryClassifier, Trainer
    from common.utils import load_labeled_data, load_word_data
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', choices=('test', 'diagnostic', 'verbose'), default='test')
    arg = parser.parse_args()

    model = BinaryClassifier()
    model.load_params('model.pkl')

    if arg.mode == 'test':
        for x in load_word_data('../../data/titles-en-test.word'):
            y = model.predict(x)
            print(f'{y}\t{" ".join(x)}')

    elif arg.mode == 'diagnostic':
        from collections import defaultdict
        from random import sample

        mistakes = defaultdict(list)
        gold = list(load_labeled_data('../../data/titles-en-test.labeled'))
        for i, x in enumerate(load_word_data('../../data/titles-en-test.word')):
            y = model.predict(x)
            t = gold[i][1]
            if y != t:
                mistakes[f'{t} -> {y}'].append(' '.join(x))
        
        for key, sentences in mistakes.items():
            print(f'{key} : {len(sentences)}')
            for line in sample(sentences, min(len(sentences), 20)):
                print(line)

    elif arg.mode == 'verbose':
        while True:
            sentence = input()
            x = sentence.replace('.', ' .').replace(',', ' ,').split()
            y = model.predict(x, verbose=True)
            print(y)