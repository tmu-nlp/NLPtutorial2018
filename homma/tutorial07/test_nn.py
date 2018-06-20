# pylint: disable=E1101
import argparse
import pickle
import numpy as np
from collections import defaultdict


def arguments_parse():
    parser = argparse.ArgumentParser(
        usage='\tNNを用いて予測するプログラム',
        description='説明',
        epilog='end',
        add_help=True,
    )
    parser.add_argument('-m', '--model', help='モデルファイル名', type=str)
    parser.add_argument('-e', '--test', help='テスト用ファイル名', type=str)
    return parser.parse_args()


def create_features_test(raw_sentence, ids):
    '''テスト用データの1行から素性を作成する'''
    phi = np.zeros(len(ids))
    words = raw_sentence.split()
    for word in words:
        if 'UNI:' + word not in ids:
            continue
        phi[ids['UNI:' + word]] += 1
    return phi


def predict_one(net, phi0):
    '''1事例を予測する'''
    phi = [0 for _ in range(len(net) + 1)]
    phi[0] = phi0
    for i in range(len(net)):
        w, b = net[i]
        phi[i + 1] = np.tanh(np.dot(w, phi[i]) + b).T
    score = phi[len(net)][0]
    return 1 if score >= 0 else -1


def test_nn(test_file, model_file, output_file):
    '''NNのテストを行う'''

    # モデルの読み込み
    with open(model_file, 'rb') as f:
        net = pickle.load(f)
        ids = pickle.load(f)

    # テスト
    with open(output_file, 'w', encoding='utf8') as f:
        for line in open(test_file, encoding='utf8'):
            raw_sentence = line.strip()
            phi = create_features_test(raw_sentence, ids)
            prediction = predict_one(net, phi)
            f.write(f'{prediction}\t{raw_sentence}\n')


if __name__ == '__main__':
    args = arguments_parse()

    model_file = args.model if args.model else r'nn_model'
    test_file = args.test if args.test else r'..\..\data\titles-en-test.word'

    test_nn(test_file, model_file, 'my_answer')


'''
../../script/grade-prediction.py ../../data/titles-en-test.labeled my_answer

Accuracy = 91.427559%
Accuracy = 91.462983% (hidden_node=4)
Accuracy = 93.623804% (lambda_=0.05, epoch=10, hidden_n=4)
Accuracy = 94.048884% (lambda_=0.03, epoch=20, hidden_n=5)
Accuracy = 93.942614% (lambda_=0.03, epoch=3, hidden_n=5)
Accuracy = 93.411265% (lambda_=0.03, epoch=1, hidden_n=5)
Accuracy = 92.490259% (lambda_=0.03, epoch=1, hidden_n=3)
Accuracy = 94.792774% (lambda_=0.03, epoch=30, hidden_n=10)
Accuracy = 94.792774% (lambda_=0.02, epoch=30, hidden_n=10)
Accuracy = 92.490259% (lambda_=0.1, epoch=3, hidden_n=5)
'''

# プログラム修正前
# Accuracy = 85.618137%
# Accuracy = 85.688983% (epoch=10)
# Accuracy = 88.310308% (lambda_=0.01)
# Accuracy = 52.072264% (lambda_=0.001)
# Accuracy = 87.141339% (lambda_*=0.99)
# Accuracy = 89.195891% (epoch=5, hidden_node=3)
# Accuracy = 87.389302% (epoch=3, hidden_node=3)
# Accuracy = 94.296847% (lambda_=0.03, epoch=10, hidden_node=5)
# Accuracy = 94.190577% (lambda_=0.03, epoch=30, hidden_node=8)
