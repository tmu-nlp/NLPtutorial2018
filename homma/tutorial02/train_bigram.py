from collections import defaultdict
import argparse


def arguments_parse():
    parser = argparse.ArgumentParser(
        usage='\tテキストデータ(input)から2-gram言語モデルを学習し，'
            + 'それを"trained_bigram"というファイル名で出力',
        description='説明',
        epilog='end',
        add_help=True,
    )
    parser.add_argument('input', help='入力ファイル名', type=str)
    return parser.parse_args()


def get_ngram(words, n):
    return zip(*[words[i:] for i in range(n)])


def train_bigram(input_file, out_file='trained_bigram'):
    counts = defaultdict(int)
    context_counts = defaultdict(int)

    # 読み込み
    for line in open(input_file, encoding='utf-8'):
        words = line.split() # split(sep=None)は先頭や末尾の空白は含まれない
        for pear in get_ngram(['<s>', *words, '</s>'], 2):
            counts[' '.join(pear)] += 1
            context_counts[pear[0]] += 1
            counts[pear[1]] += 1
            context_counts[''] += 1

    # 書き込み
    sorted_counts = sorted(counts.items())
    with open(out_file, 'w', encoding='utf-8') as f:
        for ngram, count in sorted_counts:
            probability = count/context_counts[' '.join(ngram.split()[:-1])]
            f.write(f'{ngram}\t{probability:f}\n')
    print(f'<{out_file}>に学習データを書き込みました')


if __name__ == '__main__':
    args = arguments_parse()

    train_bigram(args.input)


# 実行結果

# python train_bigram.py ..\..\data\wiki-en-train.word
# <trained_bigram>に学習データを書き込みました
