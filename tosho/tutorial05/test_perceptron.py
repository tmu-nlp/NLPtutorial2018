# python test_perceptron.py > ans.labeled
# ../../script/grade-prediction.py ../../data/titles-en-test.labeled ans.labeled

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.pardir)

    from common.binary_classifier import BinaryClassifier, Trainer
    from common.utils import load_labeled_data, load_word_data

    model = BinaryClassifier()
    model.load_params('model.pkl')

    for x in load_word_data('../../data/titles-en-test.word'):
        y = model.predict(x)
        print(f'{y}\t{" ".join(x)}')