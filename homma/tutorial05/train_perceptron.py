import argparse
from collections import defaultdict
from itertools import product
from tqdm import tqdm


def arguments_parse():
    parser = argparse.ArgumentParser(
        usage='\tパーセプトロンを用いた分類器学習プログラム',
        description='説明',
        epilog='end',
        add_help=True,
    )
    parser.add_argument('-t', '--train', help='学習用ファイル名', type=str)
    parser.add_argument('-o', '--output', help='出力ファイル名', type=str)
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


def update_weights(w, phi, y):
    '''重みwを更新する'''
    for key, value in phi.items():
        w[key] += value * y


def predict_one(w, phi):
    '''1つの事例に対する予測を返す'''
    score = 0
    for key, value in phi.items():
        if key in w:
            score += value * w[key]
    return 1 if score >= 0 else -1


def train_perceptron(epoch, train_file, output_file):
    '''パーセプトロンを用いた分類器学習'''
    w = defaultdict(float)
    for _ in tqdm(range(epoch)):
        for line in open(train_file, encoding='utf8'):
            row_label, raw_sentence = line.strip().split('\t')
            label = int(row_label)
            phi = create_features(raw_sentence)
            prediction = predict_one(w, phi)
            if prediction != label:
                update_weights(w, phi, label)
    # ファイルへ書き出し
    with open(output_file, 'w', encoding='utf8') as f:
        f.writelines(f'{k}\t{v}\n' for k, v in sorted(w.items()))


if __name__ == '__main__':
    args = arguments_parse()

    train_file = args.train if args.train else r'..\..\data\titles-en-train.labeled'
    output_file = args.output if args.output else r'perceptron_model'

    train_perceptron(28, train_file, output_file)
