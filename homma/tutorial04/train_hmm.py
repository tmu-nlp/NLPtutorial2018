import argparse
from collections import defaultdict


def arguments_parse():
    parser = argparse.ArgumentParser(
        usage='\t品詞推定のための学習プログラム',
        description='説明',
        epilog='end',
        add_help=True,
    )
    parser.add_argument('-t', '--train', help='学習用ファイル名', type=str)
    parser.add_argument('-o', '--output', help='出力ファイル名', type=str)
    return parser.parse_args()


def train_hmm(train_file, output_file):
    '''HMMで品詞推定の学習'''

    # 学習データの読み込み
    # 入力データ形式：「natural_JJ language_NN ...」
    emit = defaultdict(int)
    transition = defaultdict(int)
    context = defaultdict(int)
    for line in open(train_file, encoding='utf-8'):
        previous = '<s>'
        context[previous] += 1
        wordtags = line.split()
        for word, tag in (wt.split('_') for wt in wordtags):
            transition[f'{previous} {tag}'] += 1
            context[tag] += 1
            emit[f'{tag} {word}'] += 1
            previous = tag
        transition[f'{previous} </s>'] += 1

    # モデルの書き込み
    with open(output_file, 'w', encoding='utf8') as f:
        for key, value in transition.items():
            previous, word = key.split()
            f.write(f'T {key} {value / context[previous]}\n')
        for key, value in emit.items():
            tag, word = key.split()
            f.write(f'E {key} {value / context[tag]}\n')


if __name__ == '__main__':
    args = arguments_parse()

    train_file = args.train if args.train else '..\..\data\wiki-en-train.norm_pos'
    output_file = args.output if args.output else 'hmm_model'

    train_hmm(train_file, output_file)
