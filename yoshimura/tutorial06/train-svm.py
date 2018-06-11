# マージンを用いたオンライン学習
from collections import defaultdict
from tqdm import tqdm

epoch = 70
margin = 20
c = 0.0001


def create_features(sentence):
    phi = defaultdict(lambda: 0)
    words = sentence.split(' ')
    # 1-gram素性
    for word in words:
        phi['UNI:' + word] += 1

    # 2-gram素性
    for i in range(len(words) - 1):
        phi['BI:' + words[i] + ' ' + words[i + 1]] += 1
    return phi


def predict_one(w, phi):
    score = 0
    for name, value in phi.items():  # score = w * phi
        if name in w:
            score += value * w[name]
    if score > 0:
        return 1
    else:
        return -1


def get_val(w, phi, label, iter_, last):
    val = 0
    for name, value in phi.items():  # val = w * phi * label
        if name in w:
            val += value * getw(w, name, c, iter_, last)
    return val * label


def update_weights(w, phi, label):
    for name, value in phi.items():
        w[name] += value * label


def getw(w, name, c, iter_, last):
    if iter_ != last[name]:
        c_size = c * (iter_ - last[name])  # 溜まった更新分
        if abs(w[name]) <= c_size:
            w[name] = 0
        else:
            if w[name] >= 0:
                w[name] -= c_size
            else:
                w[name] += c_size
        last[name] = iter_
    return w[name]

if __name__ == '__main__':

    # 読み込み
    input_path = '../../data/titles-en-train.labeled'
    input_ = []  # 要素 (ラベル, 文)
    with open(input_path, 'r') as input_file:
        for line in input_file:
            columm = line.strip().split('\t')
            input_.append((int(columm[0]), columm[1]))

    # マージンを用いたオンライン学習
    w = defaultdict(lambda: 0)  # name : weight
    last = defaultdict(lambda: 0)  # name : iter(更新時)
    for _ in tqdm(range(epoch)):
        for i, (label, sentence) in enumerate(input_):
            phi = create_features(sentence)
            val = get_val(w, phi, label, i, last)
            if val <= margin:
                update_weights(w, phi, label)

    # モデル書き出し
    with open('model', 'w') as model:
        for name, weight in sorted(w.items()):
            model.write(f'{name}\t{weight}\n')

    # モデル読み込み
    w = {}  # word : weight
    with open('model', 'r') as model:
        for line in model:
            columm = line.strip().split('\t')
            w[columm[0]] = int(float(columm[1]))

    test_path = '../../data/titles-en-test.word' 

    # テストファイルを一行ずつ読み込み予測を一行ずつ出力
    with open(test_path, 'r') as test_file:
        for line in test_file:
            line = line.rstrip()
            phi = create_features(str(line))
            predicted_label = predict_one(w, phi)  # sign(w * phi)を計算
            print(predicted_label)

    # ../../script/grade-prediction.py ../../data/titles-en-test.labeled result


'''
epoch 10
1-gram            Accuracy 93.730074%
1-gram and 2-gram Accuracy 94.296847%

epoch 28
1-gram and 2-garm Accuracy 94.544810%

epoch 50
1-gram and 2-garm Accuracy 94.651080%
epoch 70
1-gram and 2-garm Accuracy 94.757350%%


'''
