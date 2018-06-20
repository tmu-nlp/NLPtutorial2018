# -*- coding: utf-8 -*-

from collections import defaultdict
import math

class HMM:
    def __init__(self):
        self.transition = defaultdict(int)
        self.emission = defaultdict(int)
        self.possible_tags = defaultdict(int)
        self.context = defaultdict(int)
        self.lambda1 = 0.95
        self.V = 1000000

    def _reset(self):
        self.__init__()

    def import_model(self, model_file, deliminator=' '):
        self._reset()
        with open(model_file, encoding='utf-8') as f:
            for line in f:
                line = line.replace('\n', '')
                items = line.split(deliminator)
                type = items[0]
                context = items[1]
                word = items[2]
                prob = items[3]

                self.possible_tags[context] = 1

                key = context + ' ' + word
                if type == "T":
                    self.transition[key] = prob
                else:
                    self.emission[key] = prob

    def train_hmm(self, _input_file, _model_file=''):
        input_file = _input_file
        model_file = _model_file
        with open(input_file, encoding='utf-8') as f_input:
            for line in f_input:
                line = line.strip()
                previous = '<s>'
                self.context[previous] += 1
                word_tags = line.split(' ')

                for word_tag in word_tags:
                    word, tag = word_tag.split('_')
                    self.transition[previous + ' ' + tag] += 1
                    self.context[tag] += 1
                    self.emission[tag + ' ' + word] += 1
                    previous = tag
                self.transition[previous+' </s>'] += 1

            if model_file != '':
                with open(model_file, 'w', encoding='utf-8') as f_model:
                    for key, value in self.transition.items():
                        previous, word = key.split(' ')
                        p = float(value) / self.context[previous]
                        f_model.write('T ' + key + ' ' + str(p) + '\n')
                    for key, value in self.emission.items():
                        tag, word = key.split()
                        p = float(value) / self.context[tag]
                        f_model.write('E ' + key + ' ' + str(p) + '\n')

    def test_hmm(self, input_file, model_file, output_file):
        def concat(_a, _b):
            return '{0} {1}'.format(_a, _b)

        self.import_model(model_file)
        f_out = open(output_file, 'w', encoding='utf-8')

        with open(input_file, 'r', encoding='utf-8') as f_input:
            # 前向きステップ
            for line in f_input:
                line = line.strip()
                words = line.split(' ')
                words.append('</s>')
                l = len(words)
                best_score = defaultdict(float)
                best_edge = defaultdict(str)
                best_score['0 <s>'] = 0
                best_edge['0 <s>'] = None

                for i in range(l):
                    for prev in self.possible_tags:
                        for next in self.possible_tags:
                            i_prev = concat(i, prev)
                            prev_next = concat(prev, next)
                            if i_prev in best_score and prev_next in self.transition:
                                next_wordsi = concat(next, words[i])
                                prob_T = float(self.transition[prev_next])
                                prob_E = self.lambda1 * float(self.emission[next_wordsi] + (1 - self.lambda1) / self.V)
                                score = best_score[i_prev] - math.log2(prob_T) - math.log2(prob_E)
                                i1_next = concat(i + 1, next)
                                if i1_next not in best_score or score > best_score[i1_next]:
                                    best_score[i1_next] = score
                                    best_edge[i1_next] = i_prev
                    for tag in self.possible_tags:
                        l_tag = concat(l, tag)
                        tag_s = concat(tag, '</s>')
                        if l_tag in best_score and tag_s in self.transition:
                            score = best_score[l_tag] - math.log2(float(self.transition[tag_s]))

                            l1_s = concat(l + 1, '</s>')
                            if l1_s not in best_score or score > best_score[l1_s]:
                                best_score[l1_s] = score
                                best_edge[l1_s] = l_tag

                # 後ろ向きステップ
                tags = []
                next_edge = best_edge[concat(l + 1, '</s>')]
                while next_edge != '0 <s>':
                    _, tag = next_edge.split(' ')
                    tags.append(tag)
                    next_edge = best_edge[next_edge]
                tags.reverse()
                print(' '.join(tags), file=output_file)


if __name__ == '__main__':
    hmm = HMM()
    input_file = '../../test/05-train-input.txt'
    model_file = '05-train-input-txt.model'
    hmm.train_hmm(input_file, model_file)
    hmm.import_model(model_file)
    hmm.train_hmm('../../data/wiki-en-train.norm', 'wiki-en-train-norm.model')
    hmm.test_hmm('../../data/wiki-en-test.norm', 'wiki-en-train-norm_pos.model', 'my_answer.pos')
