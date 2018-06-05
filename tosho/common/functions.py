import math

def sigmoid(x):
    exp = math.exp(x)
    return exp / (1 + exp)

def d_sigmoid(x):
    exp = math.exp(x)
    return exp / (1 + exp)**2

if __name__ == '__main__':
    inp = [-10, -5, -1]

    for i in inp:
        print(sigmoid(i))