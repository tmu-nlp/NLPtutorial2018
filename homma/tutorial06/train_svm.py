import argparse
import math
from collections import defaultdict
from collections import namedtuple
from itertools import product
from tqdm import tqdm


def arguments_parse():
    parser = argparse.ArgumentParser(
        usage='\tL1正則化とマージンでSVMの学習およびテストを行う',
        description='説明',
        epilog='end',
        add_help=True,
    )
    parser.add_argument('-t', '--train', help='学習用ファイル名', type=str)
    parser.add_argument('-o', '--output', help='出力ファイル名', type=str)
    parser.add_argument('-e', '--test', help='テスト用ファイル名', type=str)
    return parser.parse_args()


def create_features(raw_sentence):
    '''学習データの1行から素性を作成する'''
    phi = defaultdict(int)
    words = raw_sentence.split()
    for word in words:
        phi[f'UNI:{word}'] += 1

    for biword in zip(words[:-1], words[1:]):
        phi[f'BI:{" ".join(biword)}'] += 1

    return phi


def update_weights(w, phi, y, alpha=1):
    '''重みwを更新する'''
    for key, value in phi.items():
        w.w[key] += value * y


def eval_one(w, phi, iter_):
    '''1つの事例に対する評価値を返す'''
    score = 0
    for key, value in phi.items():
        if key in w.w:
            score += value * getw_with_l1norm(w, key, iter_)
    return score


def getw_with_l1norm(w, name, iter_, c=0.0001):
    '''L1正則化を遅延評価しつつ重みを得る'''
    if iter_ != w.last[name]:
        c_size = c * (iter_ - w.last[name])
        if abs(w.w[name]) <= c_size:
            w.w[name] = 0
        else:
            w.w[name] -= (1 if w.w[name] >= 0 else -1 ) * c_size
        w.last[name] = iter_
    return w.w[name]


def train_svm(epoch, train_file, output_file, margin=20):
    '''L1正則化とマージンで学習を行う'''
    Wlast = namedtuple('Wlast', ['w', 'last'])
    w = Wlast(defaultdict(float), defaultdict(int))
    for _ in tqdm(range(epoch)):
        for i, line in enumerate(open(train_file, encoding='utf8')):
            row_label, raw_sentence = line.strip().split('\t')
            label = int(row_label)
            phi = create_features(raw_sentence)
            score = eval_one(w, phi, i)
            val = score * label
            if val <= margin:
                update_weights(w, phi, label)

    # 本当は最後に正則化が必要

    # ファイルへ書き出し
    with open(output_file, 'w', encoding='utf8') as f:
        f.writelines(f'{k}\t{v}\n' for k, v in sorted(w.w.items()))


def predict_one(w, phi):
    '''1事例を予測する'''
    score = sum(value * w[key] for key, value in phi.items())
    return 1 if score >= 0 else -1

def test_svm(test_file, model_file, output_file):
    '''SVMのテストを行う'''

    # モデルの読み込み
    w = defaultdict(float)
    for line in open(model_file, encoding='utf8'):
        key, raw_value = line.strip().split('\t')
        w[key] = float(raw_value)

    # テスト
    with open(output_file, 'w', encoding='utf8') as f:
        for line in open(test_file, encoding='utf8'):
            raw_sentence = line.strip()
            phi = create_features(raw_sentence)
            prediction = predict_one(w, phi)
            f.write(f'{prediction}\t{raw_sentence}\n')


if __name__ == '__main__':
    args = arguments_parse()

    train_file = args.train if args.train else r'..\..\data\titles-en-train.labeled'
    model_file = args.output if args.output else r'perceptron_model'
    test_file = args.test if args.test else r'..\..\data\titles-en-test.word'

    train_svm(10, train_file, model_file)
    test_svm(test_file, model_file, 'my_answer')


'''
../../script/grade-prediction.py ../../data/titles-en-test.labeled my_answer

Accuracy = 93.552958% (1-gram, epoch 10, this(svm))
Accuracy = 93.446688% (1-gram, epoch 10, perceptron)

Accuracy = 94.226001% (2-gram, epoch 10, this(svm))
'''
