# 重みを読み込み、予測を１行ずつ出力
from train_perceptron import create_features
from train_perceptron import predict_one

w = {} # word : weight
with open('model','r') as model:
    for line in model:
        columm= line.strip().split('\t')
        w[columm[0]] =  int(columm[1])

test_path = '../../data/titles-en-test.word' # テスト用
# test_path = 'test.txt'

with open(test_path,'r') as test_file:
    for line in test_file:
        line = line.rstrip()
        phi = create_features(str(line)) 
        predicted_label = predict_one(w, phi) # sign(w * phi)を計算
        print(predicted_label)


# ../../script/grade-prediction.py data/titles-en-test.labeled result

        