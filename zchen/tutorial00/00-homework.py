import sys
from collections import defaultdict

count_dict = defaultdict(lambda : 0)

with open(sys.argv[1], 'r') as fr:
    for line in fr:
        for token in line.strip().split(" "):
            count_dict[token] += 1

for pair in count_dict.items():
    print( *pair )
