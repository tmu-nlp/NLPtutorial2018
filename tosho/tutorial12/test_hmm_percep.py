import sys, os
from sklearn.externals import joblib
from train_hmm_percep import hmm_viterbi

def main():
    model_path = sys.argv[1]
    w, transition, tags = joblib.load(model_path)

    for X in load_train_data(sys.stdin):
        Y = hmm_viterbi(w, X, transition, tags)
        print(*Y)

def load_train_data(doc):
    for line in doc:
        yield line.strip().split()

if __name__ == '__main__':
    main()
    # import cProfile
    # cProfile.run('main()')