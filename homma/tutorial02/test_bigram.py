from collections import defaultdict
import argparse
import math

def arguments_parse():
    parser = argparse.ArgumentParser(
        usage='\t学習済みの2-gram言語モデルを用いて評価データのエントロピーを計算',
        description='説明',
        epilog='end',
        add_help=True,
    )
    parser.add_argument('input', help='入力ファイル名', type=str)
    parser.add_argument('-t', '--trained', help='学習データファイル名', type=str)
    return parser.parse_args()


def get_ngram(words, n):
    return zip(*[words[i:] for i in range(n)])


def test_bigram(test_file, trained='trained_bigram', lambda1=0.3, lambda2=0.3):
    V = 1e6
    W = 0
    H = 0

    # モデルの読み込み
    probs = defaultdict(float)
    for line in open(trained, encoding='utf-8'):
        splited = line.strip('\n').split('\t')
        probs[splited[0]] = float(splited[1])

    # 評価と結果表示
    for line in open(test_file, encoding='utf-8'):
        words = line.split()
        words.append('</s>')
        W += len(words)
        for pear in get_ngram(['<s>', *words, '</s>'], 2):
            p1 = lambda1 * probs[' '.join(pear[-1])] + (1 - lambda1) / V
            p2 = lambda2 * probs[' '.join(pear)] + (1 - lambda2) * p1
            H -= math.log2(p2)
    print(f'entropy = {H/W}')
    return H/W


if __name__ == '__main__':
    args = arguments_parse()

    if args.trained:
        test_bigram(args.input, args.trained)
    else:
        test_bigram(args.input)


# 実行結果

# python test_bigram.py ..\..\data\wiki-en-test.word
# entropy = 14.248907725927946
