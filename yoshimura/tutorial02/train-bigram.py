import sys
from collections import defaultdict

counts = defaultdict(lambda:0)
context_counts = defaultdict(lambda:0)

with open(sys.argv[1],'r',encoding = 'utf-8') as train_file:
    for line in train_file:
        line = line.strip()
        words = line.split(" ")
        words.insert(0, '<s>')
        words.append('</s>')

        for i in range(1, len(words)): # <s>の後から
            # 2-gramの分子と分母を加算
            counts[f'{words[i-1]} {words[i]}'] += 1
            context_counts[f'{words[i-1]}'] += 1
            # 1-gramの分子と分母を加算
            counts[f'{words[i]}'] += 1
            context_counts[''] += 1

        context_counts[f'{words[len(words) - 1]}'] += 1 # </s>をカウント

with open('model','w',encoding = 'utf-8') as model_file:
    for ngram, count in sorted(counts.items()):
        words = ngram.split(' ') # 文字列を1文字ずつのリストにする
        words.pop()
        context = ''.join(words)
        probability = counts[ngram]/context_counts[context]
        model_file.write(ngram + "\t" + str(probability) + '\n')