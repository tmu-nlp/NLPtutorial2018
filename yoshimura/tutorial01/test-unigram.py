import math
import sys
lam = 0.95
lam_unknown = 1 - lam
V = 10**6 #未知語を含む語彙数
W = 0 #testの単語数
H = 0
unknown = 0 # 未知語の数

# モデル読み込み
probabilities = {}
with open(sys.argv[1],"r") as model_file:
    for line in model_file:
        line = line.strip()
        model = line.split("\t")
        probabilities[model[0]] = float(model[1])


# 評価と結果表示
with open(sys.argv[2],"r") as test_file:
    for line in test_file:
        line = line.strip()
        words = line.split(" ")
        words.append("</s>")
        for word in words:
            W += 1 # test dataの単語数カウント
            if word in probabilities:
                P_ml = probabilities[word]
                P = lam * P_ml + lam_unknown / V #未知語対応
            else:
                P = lam_unknown / V
                unknown += 1 
            # print(f"P({word})\t = {lam} * {P_ml} + {lam_unknown:.2f} / {V} = {P}")
            H += -math.log(P,2)


print(f"unknown : {unknown}")
print(f"W : {W}")
print(" ")
print(f"entropy = {H/W}")
print(f"coverage = {(W - unknown)/W}")




    
        
        
