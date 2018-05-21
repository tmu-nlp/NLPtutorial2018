# python word_segmentation.py ../../data/wiki-ja-train.word model.txt ../../data/wiki-ja-test.txt my_answer.word
# ../../script/gradews.pl ../../data/wiki-ja-test.word my_answer.word

import sys
import math
from collections import defaultdict


def train_unigram(input_file, model_file):

    counts = defaultdict(int)
    total_count = 0

    with open(input_file, 'r') as input_text, open(model_file, "w") as output_text:
        for line in input_text:
            words = line.split()
            words.append("</s>")
            for word in words:
                counts[word] += 1
                total_count += 1

        for word, count in counts.items():
            probability = float(counts[word]/total_count)
            output_text.write(word + "\t" + str(probability) + "\n")


def vitervi(model_file, test_file, output_file):
    probabilities = defaultdict(int)
    with open(model_file, 'r') as model_text, open(test_file, 'r') as test_text, open(output_file, 'w') as output_text:
        for line in model_text:
            line = line.split()
            probabilities[line[0]] = float(line[1])

        lambda_1 = 0.95
        lambda_unk = 1 - lambda_1
        V = 1000000
        for line in test_text:
            # 前向きステップ
            best_edge = dict()
            best_score = dict()
            best_edge[0] = 'NULL'
            best_score[0] = 0
            line = line.strip() # remove newline
            for word_end in range(1, len(line)+1):
                best_score[word_end] = 10**10
                for word_begin in range(0, len(line)):
                    word = line[word_begin:word_end]
                    if word in probabilities or len(word) == 1:
                        prob = lambda_1 * probabilities[word] + lambda_unk / V
                        my_score = best_score[word_begin] - math.log(prob, 2)
                        if my_score < best_score[word_end]:
                            best_score[word_end] = my_score
                            best_edge[word_end] = (word_begin, word_end)

            #プリントしてみる
            print(best_edge)
            print(best_score)

            # 後向きステップ
            words = []
            next_edge = best_edge[len(best_edge)-1] # next_edge=(n, n_1)
            while next_edge != 'NULL':
                word = line[next_edge[0]:next_edge[1]]
                words.append(word)
                next_edge = best_edge[next_edge[0]] # next_edge=(n_-1,n)
            words.reverse()
            print(' '.join(words))
            print('##########################################')
            output_text.write(' '.join(words) + '\n')



if __name__ == '__main__':
    argvs = sys.argv
    argc = len(argvs)
    if argc != 5:
        print('python word_segmentation.py MODEL_SOURCE MODEL_OUTPUT INPUT_TEXT RESULT_OUTPUT')
    else:
        train_unigram(argvs[1], argvs[2])
        vitervi(argvs[2], argvs[3], argvs[4])