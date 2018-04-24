import sys
from collections import defaultdict
import random

def load_file(src, includes_eos=False):
    '''
    EOS(</s>)を考慮するかどうか指定して、
    トレーニングデータを読み込みます。
    '''
    if includes_eos:
        return map(lambda line: line.strip() + ' </s>', src)
    else:
        return map(lambda line: line.strip(), src)


def count_words(target_filename, verbose=False, includes_eos=False):
    ret = defaultdict(lambda: 0)
    target_file = open(target_filename, 'r')

    for line in load_file(target_file, includes_eos):
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
