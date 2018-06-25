# pylint: disable=E1101
import argparse
import pickle
import numpy as np
from collections import defaultdict


def arguments_parse():
    parser = argparse.ArgumentParser(
        usage='\tRNNを用いて品詞推定プログラム',
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
../../script/grade-prediction.py ../../data/titles-en-test

'''

