# パーセプトロを用いた分類学習
from collections import defaultdict

epoch = 150

def create_features(sentence):
    phi = defaultdict(lambda:0)
    words = sentence.split(' ')
    # 1-gram素性
    for word in words:
        phi['UNI:' + word] += 1

    # 2-gram素性
    # for i in range(len(words) - 1):
    #     phi['BI:' + words[i] + words[i + 1]] += 1
    return phi

def predict_one(w, phi):
    score = 0
    for name, value in phi.items(): # score = w * phi
        if name in w:
            score += value * w[name]
    if score >= 0:
        return 1
    else:
        return -1

def update_weights(w, phi, label):
    for name, value in phi.items():
        w[name] += value * label

# 読み込み
# input_path = '../../test/03-train-input.txt' # テスト用
input_path = '../../data/titles-en-train.labeled'
input_ = [] # 要素 (ラベル, 文)
with open(input_path, 'r') as input_file:
    for line in input_file:
        columm= line.strip().split('\t')
        input_.append((int(columm[0]), columm[1]))

# オンライン学習
w = defaultdict(lambda:0) # name : weight
for i in range(epoch):
    for label, sentence in input_:
        phi = create_features(sentence)
        predicted_label = predict_one(w, phi)
        if predicted_label != label:
            update_weights(w, phi, label)

with open('model','w') as model:
    for name, weight in sorted(w.items()):
        model.write(f'{name}\t{weight}\n')

# epoch 10  1-gram 93.446688
# epoch 50  1-gram 93.269571
# epoch 100 1-gram 93.552958
# epoch 150 1-gram 93.552958
# epoch 10  2-gram 92.4194
# epoch 50  2-gram 92.773645