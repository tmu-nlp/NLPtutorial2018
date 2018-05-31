import argparse
from collections import defaultdict

from train_perceptron import create_features, predict_one


def arguments_parse():
    parser = argparse.ArgumentParser(
        usage='\tパーセプトロンを用いた分類器による分類テストプログラム',
        description='説明',
        epilog='end',
        add_help=True,
    )
    parser.add_argument('-m', '--model', help='モデルファイル名', type=str)
    parser.add_argument('-t', '--test', help='テスト用ファイル名', type=str)
    parser.add_argument('-o', '--output', help='出力ファイル名', type=str)
    return parser.parse_args()


def test_perceptron(test_file, model_file, output_file):
    '''パーセプトロンを用いた分類器テスト'''

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

    model_file = args.model if args.model else r'perceptron_model'
    test_file = args.test if args.test else r'..\..\data\titles-en-test.word'
    output_file = args.output if args.output else r'my_answer'

    test_perceptron(test_file, model_file, output_file)


'''
../../script/grade-prediction.py ../../data/titles-en-test.labeled my_answer

Accuracy = 93.871768%
'''


# Accuracy = 90.967056% 1
# Accuracy = 91.781792% 2
# Accuracy = 92.773645% 3
# Accuracy = 90.825363% 4
# Accuracy = 91.852639% 5
# Accuracy = 91.781792% 6
# Accuracy = 92.738222% 7
# Accuracy = 91.569253% 8
# Accuracy = 92.383989% 9
# Accuracy = 93.446688% 10
# Accuracy = 93.304995% 11
# Accuracy = 92.915338% 12
# Accuracy = 93.269571% 13
# Accuracy = 93.623804% 14
# Accuracy = 93.057032% 15
# Accuracy = 93.198725% 16
# Accuracy = 93.588381% 17
# Accuracy = 93.021608% 18
# Accuracy = 93.304995% 19
# Accuracy = 93.234148% 20
# Accuracy = 92.879915% 21
# Accuracy = 93.517535% 22
# Accuracy = 93.482111% 23
# Accuracy = 92.986185% 24
# Accuracy = 93.092455% 25
# Accuracy = 93.092455% 26
# Accuracy = 93.588381% 27
# Accuracy = 93.871768% 28
# Accuracy = 92.809068% 29
# Accuracy = 92.950762% 30
# Accuracy = 93.800921% 31
# Accuracy = 93.304995% 32
# Accuracy = 93.234148% 33
# Accuracy = 93.694651% 34
# Accuracy = 93.517535% 35
# Accuracy = 93.269571% 36
# Accuracy = 93.304995% 37
# Accuracy = 93.198725% 38
# Accuracy = 93.730074% 39
# Accuracy = 93.446688% 40
# Accuracy = 92.986185% 41
# Accuracy = 93.552958% 42
# Accuracy = 93.588381% 43
# Accuracy = 93.269571% 44
# Accuracy = 92.561105% 45
# Accuracy = 93.588381% 46
# Accuracy = 93.446688% 47
# Accuracy = 93.446688% 48
# Accuracy = 93.623804% 49
# Accuracy = 93.269571% 50
# Accuracy = 93.552958% < 50
