import sys
from collections import defaultdict

def create_features(x):
    phi = defaultdict(int)
    words = x.split(" ")
    for word in words:
            phi["UNI:"+word] += 1
    return phi

def predict_one(w,phi):
    score = 0
    for name,value in phi.items():
        if name in w:
            score += value * w[name]
    if score >= 0:
        return 1
    else:
        return -1

def update_weights(w,phi,y):
    for name, value in phi.items():
        w[name] += int(value) * int(y)

if __name__ == '__main__':
    w = defaultdict(int)
    with open(sys.argv[1], 'r') as fin, open("model.txt", 'w') as fout:
        for line in fin:
            splited_line = line.strip().split("\t")
            x = splited_line[1]
            y = splited_line[0]
            phi = create_features(x)
            if predict_one(w, phi) != y:
                update_weights(w, phi, y)
        for key, value in w.items():
            fout.write(f"{key}\t{value}\n")
