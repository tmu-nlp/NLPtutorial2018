'''
python test_sr.py <../../data/mstparser-en-test.dep >ans.dep
python ../../script/grade-dep.py ../../data/mstparser-en-test.dep ans.dep
'''

import pickle as pkl
import train_sr as sr

def main():
    W = pkl.load(open(sr.WEIGHT_PATH, 'rb'))
    data = sr.load_data()

    for sentence in data:
        heads = sr.shift_reduce(sentence, W, mode='test')
        for token, head in zip(sentence, heads[1:]):
            token.head = head
            token.attrs[6] = head
            print('\t'.join(map(str,token.attrs)))
        print()

if __name__ == '__main__':
    main()
    # import cProfile
    # cProfile.run('main()')