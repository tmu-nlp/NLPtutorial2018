#word counter sys[1] = file dir, sys[2] = 1(all), 2(top 10), 3(number of type), other(all)

import sys
from collections import Counter

def word_count(f_name, mode = 1):
    word_list = Counter()
    with open(f_name) as f:
        for line in f:
            word_list += Counter(line.split())
    if mode == 1:
        for word, n in word_list.most_common():
            print(word, n)
    elif mode == 2:
        for word, n in word_list.most_common(10):
            print(word, n)
    elif mode == 3:
        print('異なり数：{}'.format(len(word_list)))



if __name__ == '__main__':
    input_file = sys.argv[1]
    try:
        mode = int(sys.argv[2])
        if mode not in [1,2,3]:
            mode = 1
    except:
        mode = 1
    word_count(input_file, mode)

