import sys
from collections import defaultdict

counts = defaultdict(lambda: 0)

word_file = open(sys.argv[1], "r")

for line in word_file:
    line = line.strip()
    words = line.split()

    for w in words:
        counts[w] += 1

if len(sys.argv) > 2:
    if sys.argv[2] == "1":　#単語の異なり数
        print(f'単語の異なり数 = {len(counts)}')
    elif sys.argv[2] == "2": #出現頻度上位10単語
        i = 0
        for foo, bar in sorted(counts.items(), key=lambda x: -x[1]):
            if i < 10:
                print(f'{foo} {bar}')
                i += 1
            else:
                break
else: #全単語の頻度(アルファベット順)
    for foo, bar in sorted(counts.items()):
        print(f'{foo} {bar}')

