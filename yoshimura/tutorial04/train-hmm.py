from collections import defaultdict

emit = defaultdict(lambda:0)
transition = defaultdict(lambda:0)
context = defaultdict(lambda:0)

train_path = '../../data/wiki-en-train.norm_pos'
# train_path = '../../test/05-train-input.txt' # テスト用ファイルパス

with open(train_path, 'r') as train_file:
    for line in train_file:
        line = line.rstrip()
        previous = '<s>'
        context[previous] += 1
        word_tags = line.split(' ')
        for word_tag in word_tags:
            word, tag = word_tag.split('_')
            transition[previous + ' ' + tag] += 1 # 遷移を数え上げる
            context[tag] += 1 # 文脈を数え上げる
            emit[tag + ' ' + word] += 1 # 生成を数え上げる
            previous = tag
        transition[previous + ' ' + '</s>'] += 1

    # 遷移を出力
    for key, value in sorted(transition.items()):
        previous, word = key.split(' ')
        print(f'T {key} {value/context[previous]}')
    # 生成確率を出力
    for key, value in sorted(emit.items()):
        previous, word = key.split(' ')
        print(f'E {key} {value/context[previous]}')
            
            
