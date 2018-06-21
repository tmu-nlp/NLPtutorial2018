# python test_perceptron.py > ans.labeled
# ../../script/grade-prediction.py ../../data/titles-en-test.labeled ans.labeled
# ../../script/grade-prediction.py ../../data/titles-en-test.labeled ans_unsemble.labeled

''' 
python train_svm.py
python test_svm.py > ans.labeled
../../script/grade-prediction.py ../../data/titles-en-test.labeled ans.labeled
'''

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.pardir)

    from common.svm import BinaryClassifier, Trainer
    from common.utils import load_labeled_data, load_word_data
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', choices=('test', 'ensemble', 'diag', 'verbose', 'param'), default='test')
    arg = parser.parse_args()

    if arg.mode == 'test':
        model = BinaryClassifier()
        model.load_params('model.pkl')
        
        for x in load_word_data('../../data/titles-en-test.word'):
            y = model.predict(x)
            print(f'{y}\t{" ".join(x)}')
    
    elif arg.mode == 'ensemble':
        def load_model(file_name):
            model = BinaryClassifier()
            model.load_params(file_name)
            return model
        
        models = [load_model('model_' + str(i) + '.pkl') for i in range(1, 6)]
        for x in load_word_data('../../data/titles-en-test.word'):
            y = [model.predict(x) for model in models]
            y = np.sign(sum(y))
            print(f'{y}\t{" ".join(x)}')

    elif arg.mode == 'diag':
        from collections import defaultdict
        from random import sample

        model = BinaryClassifier()
        model.load_params('model.pkl')
        
        for name, value in sorted(model.params.items(), key=lambda i: -abs(i[1]))[:10]:
            print(f'{name} : {value}')

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
    
    elif arg.mode == 'param':
        model = BinaryClassifier()
        model.load_params('model.pkl')
        while True:
            p = input()
            for name, value in filter(lambda item: item[0].startswith(p), model.params.items()):
                print(f'{name} : {value}')
