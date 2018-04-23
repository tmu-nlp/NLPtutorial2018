# -*- coding: utf-8 -*-
import re


class TrainUnigram():
    def __init__(self):
        pass

    def train_model(self, input_file, model_file):
        counts = {}
        total_count = 0

        with open(input_file, encoding='utf-8')as f:
            for line in f:
                line = line.lower()
                line = re.sub(r'[\.,\^\?!\+-;\'\"`~=\[\]\{\}\$%&\\\*]', '', line)
                line = re.sub(r'[\r\n]', '', line)
                line = re.sub(r' +', ' ', line)
                words = line.split(' ')
                words.append('</s>')
                for word in words:
                    if word == '':
                        continue
                    if word in counts:
                        counts[word] += 1
                    else:
                        counts[word] = 1
                    total_count += 1
            print('keys:{0}, total:{1}'.format(len(counts.keys()), total_count))

        with open(model_file, 'w', encoding='utf-8') as f:
            for key in counts.keys():
                probability = counts[key] / float(total_count)
                f.write('{0},{1}\n'.format(key, probability))
                # print('key:{0}, prob:{1}'.format(key, probability))

if __name__ == '__main__':
    trainer = TrainUnigram()
    trainer.train_model('../../data/wiki-en-train.word', 'model.txt')