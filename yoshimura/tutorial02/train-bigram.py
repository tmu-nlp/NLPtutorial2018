from collections import defaultdict

counts = defaultdict(lambda:0) # ngram : 回数
context_counts = defaultdict(lambda:0) # bigramの先頭の単語 : 回数
type_counts = defaultdict(set) # bigramの先頭の単語 : bigramの後ろの単語の集合
unigram_counts = defaultdict(lambda:0) # unigramn : 回数

path_train = '../../data/wiki-en-train.word'
with open(path_train,'r',encoding = 'utf-8') as train_file:
    for line in train_file:
        line = line.strip()
        words = line.split(" ")
        words.insert(0, '<s>')
        words.append('</s>')

        for i in range(1, len(words)): # <s>の後から
            # 2-gramの分子と分母をカウント
            counts[words[i-1] + ' ' + words[i]] += 1
            context_counts[words[i-1]] += 1
            # 1-gramの分子と分母をカウント
            counts[words[i]] += 1
            context_counts[''] += 1 # 全体の単語数カウント

            # w[i]の異なり数
            type_counts[words[i-1]].add(words[i]) 
            # unigramの述べと異なりをカウント
            unigram_counts[words[i]] += 1

with open('model','w',encoding = 'utf-8') as model_file:

    # unigramのlambdaを計算
    unigram_token = sum(unigram_counts.values())
    unigram_type = len(unigram_counts) 
    lam = unigram_type/(unigram_token + unigram_type)

    for ngram, count in sorted(counts.items()):
        words = ngram.split(' ') # bigramを単語に分割
        words.pop() # 着目単語のみにする
        context = ''.join(words) # 文字列に変換
        probability = count/context_counts[context] 

        # Witten-Bell平滑化によってlambdaを選ぶ
        u = len(type_counts[context]) # w[i-1]の後に続く単語の異なり数
        witten_lambda = 1 - (u/(u + context_counts[context]))

        if len(ngram.split(' ')) > 1:
            model_file.write(ngram + '\t' + str(probability) + '\t' + str(witten_lambda) + '\n')
        else:
            model_file.write(ngram + '\t' + str(probability) + '\t' + str(lam) + '\n')
            
