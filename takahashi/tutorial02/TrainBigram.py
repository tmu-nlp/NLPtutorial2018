# -*- coding: utf-8 -*-
import re


class TrainBigram():
    def __init__(self):
        pass

    def count_up(self, dictionary, key):
        if key in dictionary:
            dictionary[key] += 1
        else:
            dictionary[key] = 1
        return dictionary

    def train_model(self, input_file, model_file):
        counts = {}
        context_counts = {}
        total_counts = 0

        with open(input_file, encoding='utf-8')as f:
            for line in f:
                line = line.lower()
                line = re.sub(r'[\.,\^\?!\+-;\'\"`~=\[\]\{\}\$%&\\\*]', '', line)
                line = re.sub(r'[\r\n]', '', line)
                line = re.sub(r' +', ' ', line)
                words = line.split(' ')
                words.insert(0, '<s>')
                words.append('</s>')
                for index in range(1, len(words)):
                    if words[index] == '' or words[index - 1] == '':
                        continue

                    key = words[index - 1] + ' ' + words[index]
                    counts = self.count_up(counts, key)
                    key = words[index]
                    counts = self.count_up(counts, key)

                    key = words[index - 1]
                    context_counts = self.count_up(context_counts, key)


                    total_counts += 1

                key = words[-1]
                context_counts = self.count_up(context_counts, key)

            print('context_counts keys:{0}, counts keys:{1}, total:{2}'.format(len(context_counts.keys()), len(counts.keys()), total_counts))

        with open(model_file, 'w', encoding='utf-8') as f:
            for ngram in counts.keys():
                if ngram.find(' ') != -1:
                    words = ngram.split(' ')
                    context = words[0]
                    probability = counts[ngram] / float(context_counts[context])
                else:
                    probability = counts[ngram] / float(total_counts)
                # print('key:{0}, prob:{1}'.format(ngram, probability))
                f.write('{0},{1}\n'.format(ngram, probability))

if __name__ == '__main__':
    trainer = TrainBigram()
    trainer.train_model('../../test/02-train-input.txt', 'test_model.txt')