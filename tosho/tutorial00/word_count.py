import sys
from collections import defaultdict
import random

def count_words(target_filename, verbose=False):
    ret = defaultdict(lambda: 0)
    target_file = open(target_filename, 'r')

    for line in target_file:
        line = line.strip()
        if len(line) != 0:
            words = line.split(' ')
            for word in words:
                if len(word) != 0:
                    ret[word] += 1

    target_file.close()

    if verbose:
        for word, count in ret.items():
            print('{0} : {1}'.format(word, count))

    return ret

if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = None

    if filename != None:
        ret = count_words(filename)
        for word, count in ret.items():
            print('{0} {1}'.format(word, count))
        print('%d words' % (len(ret)))
