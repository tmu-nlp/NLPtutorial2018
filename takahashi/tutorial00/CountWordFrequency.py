# -*- coding: utf-8 -*-

def count_word_frequency(f):
    counts = {}
    for line in f:
        line = line.replace('\n', '')
        words = line.split(' ')

        for w in words:
            if w in counts:
                counts[w] += 1
            else:
                counts[w] = 1

    print(counts)
    return counts

def get_answer(f):
    counts = {}
    for line in f:
        line = line.replace('\n', '')
        pair = line.split('\t')

        counts[pair[0]] = int(pair[1])
    print(counts)
    return counts

def test_same_as_result(test, answer):
    if test.keys() != answer.keys():
        print("test FAIL!!!", test.keys(), answer.keys())
        return 0
    for key in answer.keys():
        if test[key] != answer[key]:
            print("test FAIL!!!", key, test[key], answer[key])
            return 0
    print('test PASS!!!!')
    return 1

with open('../../test/00-input.txt', encoding='utf-8')as f:
    test_result = count_word_frequency(f)
with open('../../test/00-answer.txt', encoding='utf-8')as f:
    answer = get_answer(f)

test_same_as_result(test_result, answer)

with open('../../data//wiki-en-train.word', encoding='utf-8')as f:
    count_word_frequency(f)