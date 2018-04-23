from collections import defaultdict
import sys

word_dict = defaultdict(lambda: 0)

def main(input_file):
    with open(input_file, 'r') as input_f:
        for line in input_f:
            words = line.strip().split()
        
            for word in words:
                # 文頭で一文字目が大文字になってるようなやつも小文字と一緒に数えたいとき
                # word = word.lower()
                word_dict[word] += 1

        # 異なり数
        print('{}: {}'.format('異なり数', len(word_dict)))

        # keyを出現数でソート。マイナスをつけると降順になる。後ろのは何番目まで出すか決めるやつ
        for key, value in sorted(word_dict.items(), key=lambda x: -x[1])[:10]:
            print('{} {}'.format(key, value))


if __name__ == '__main__':
    argvs = sys.argv
    argc = len(argvs)
    if argc != 2:
        print('引数に"INPUT_FILE"をつけて使います')
    else:
        main(argvs[1])