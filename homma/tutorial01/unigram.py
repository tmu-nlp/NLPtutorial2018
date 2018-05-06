from collections import defaultdict
import sys
import argparse
import math

separator = '\t'


def arguments_parse():
    parser = argparse.ArgumentParser(
        usage='\t学習モード：テキストデータから1-gram言語モデルを学習し，それを"trained"というファイル名で出力' +
        '\n\tテストモード：学習済みの1-gram言語モデルを用いてテキストデータのエントロピーとカバレージを計算',
        description='説明',
        epilog='end',
        add_help=True,
    )
    parser.add_argument('input', help='入力ファイル名', type=str)
    parser.add_argument('-t', '--trained', help='学習データファイル名', type=str)
    parser.add_argument(
        '-m', '--mode', help='テスト:0 学習:1 省略時はテスト', type=int, choices=[0, 1])
    return parser.parse_args()


def train_unigram(input_file, out_file='trained.csv'):
    counts = defaultdict(lambda: 0)

    # 読み込み
    for line in open(input_file, encoding='utf-8'):
        line = line.strip()
        words = line.split(" ")
        for word in words:
            counts[word] += 1
        counts['</s>'] += 1
    word_sum = sum(counts.values())
    print('総単語数', word_sum)

    # 書き込み
    sorted_counts = sorted(counts.items())
    with open(out_file, 'w', encoding='utf-8') as f:
        for key, value in sorted_counts:
            f.write(f'{key}{separator}{value/word_sum:f}\n')
    print(f'<{out_file}>に学習データを書き込みました')


def test_unigram(test_file, trained='trained.csv'):
    lambda1 = 0.95
    unk = 1 - lambda1
    V = 1e6
    W = 0
    H = 0

    # モデルの読み込み
    probabilities = {}
    for line in open(trained, encoding='utf-8'):
        splited = line.strip('\n').split(separator)
        probabilities[splited[0]] = splited[1]

    # 評価と結果表示
    for line in open(test_file, encoding='utf-8'):
        words = line.strip().split(' ')
        words.append('</s>')
        W += len(words)
        unk_cnt = 0
        for w in words:
            p = unk / V
            if w in probabilities:
                p += lambda1 * float(probabilities[w])
            else:
                unk_cnt += 1
            H += -math.log2(p)
    print(f'entropy = {H/W}')
    print(f'coverage = {(W-unk_cnt)/W}')


if __name__ == '__main__':
    args = arguments_parse()

    if not args.trained:
        if not args.mode:
            test_unigram(args.input)
        elif args.mode == 1:
            train_unigram(args.input)
    else:
        if not args.mode:
            test_unigram(args.input, args.trained)
        elif args.mode == 1:
            train_unigram(args.input, args.trained)


# 実行結果

# python unigram.py ..\..\data\wiki-en-train.word -m1
# 総単語数 35842
# <trained.csv>に学習データを書き込みました

# python unigram.py ..\..\data\wiki-en-test.word
# entropy = 10.526656347101143
# coverage = 1.0
