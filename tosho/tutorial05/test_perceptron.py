# python test_perceptron.py > ans.labeled
# python test_perceptron.py > anx.labeled
# ../../script/grade-prediction.py ../../data/titles-en-test.labeled ans.labeled

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.pardir)

    from common.binary_classifier import BinaryClassifier, Trainer
    from common.utils import load_labeled_data, load_word_data
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', choices=('test', 'unsemble', 'diagnostic', 'verbose'), default='test')
    arg = parser.parse_args()

    if arg.mode == 'test':
        model = BinaryClassifier()
        model.load_params('model.pkl')
        
        for x in load_word_data('../../data/titles-en-test.word'):
            y = model.predict(x)
            print(f'{y}\t{" ".join(x)}')
    
    elif arg.mode == 'unsemble':
        def load_model(file_name):
            model = BinaryClassifier()
            model.load_params(file_name)
            return model
        
        models = [load_model('model_' + str(i) + '.pkl') for i in range(1, 6)]
        for x in load_word_data('../../data/titles-en-test.word'):
            y = [model.predict(x) for model in models]
            y = np.sign(sum(y))
            print(f'{y}\t{" ".join(x)}')

    elif arg.mode == 'diagnostic':
        from collections import defaultdict
        from random import sample

        model = BinaryClassifier()
        model.load_params('model.pkl')

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
        
        model = BinaryClassifier()
        model.load_params('model.pkl')

        while True:
            sentence = input()
            x = sentence.replace('.', ' .').replace(',', ' ,').split()
            y = model.predict(x, verbose=True)