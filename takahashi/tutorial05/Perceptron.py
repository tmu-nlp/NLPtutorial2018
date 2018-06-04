# -*- coding: utf-8 -*-
from collections import defaultdict
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.linear_model import Perceptron as ppn
from sklearn.metrics import accuracy_score
import Stemmer
from nltk import stem, tokenize
from scipy.sparse import hstack, vstack



class Perceptron:
    def __init__(self):
        self.model = defaultdict(int)
        self.iterations = 100
        self.tfidf = None

    def import_model(self, _model_file):
        with open(_model_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.replace('\n', '')
                key, value = line.split('\t')
                self.model[key] = float(value)

    def save_model(self, _model_file):
        with open(_model_file, 'w', encoding='utf-8') as f:
            for key, value in self.model.items():
                f.write('{0}\t{1:.6f}\n'.format(key, value))

    def reset_model(self):
        self.model = defaultdict(int)

    def train(self, _input_file, _model_file=''):
        with open(_input_file, 'r', encoding='utf-8') as f:
            for itr in range(self.iterations):
                for line in f:
                    line = line.strip()
                    y, x = line.split('\t')
                    y = float(y)
                    phi = self.create_features(x)
                    y_prime = self.predict_one(phi)
                    if y_prime != y:
                        self.update_weights(phi, y)

        if _model_file != '':
            self.save_model(_model_file)

    def predict_all(self, _model_file, _input_file, _output_file):
        self.import_model(_model_file)

        f_out = open(_output_file, 'w', encoding='utf-8')

        with open(_input_file, 'r', encoding='utf-8') as f:
            for line in f:
                phi = self.create_features(line)
                y_prime = self.predict_one(phi)
                ret = '{0}\t{1}'.format(y_prime, line)
                f_out.write(ret)
        f_out.close()

    def predict_one(self, phi):
        score = 0.0
        for name, value in phi.items():
            if name in self.model:
                score += value * self.model[name]
        if score >= 0:
            return 1
        else:
            return -1

    def update_weights(self, phi, y):
        for name, value in phi.items():
            self.model[name] += value * y

    def create_features(self, x):
        if self.tfidf is not None:
            return self.tfidf_features(x)
        return self.default_features(x)

    def default_features(self, x):
        phi = defaultdict(int)
        words = x.strip(). split(' ')
        for word in words:
            phi['UNI:' + word] += 1
        return phi

    def tfidf_vectorize(self, _input_file, _tfidf_model_file, _use_model_file=False):
        if _use_model_file:
            with open(_tfidf_model_file, 'rb') as f:
                self.tfidf = pickle.load(f)
        else:
            tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=1, max_features=3000, norm='l2')
            self.tfidf = tfidf_vectorizer.fit(_input_file)
            with open(_tfidf_model_file, 'wb') as f:
                pickle.dump(self.tfidf, f)
        return self.tfidf

    def tfidf_features(self, x):
        return self.tfidf.transform(x)

english_stemmer = Stemmer.Stemmer('en')
class StemmedTfidVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: english_stemmer.stemWords(analyzer(doc))



if __name__ == '__main__':
    perceptron = Perceptron()

    input_file = '../../test/03-train-input.txt'
    model_file = '03-train-input-txt.model'
    perceptron.train(input_file, model_file)
    perceptron.reset_model()

    # Accuracy = 90.967056%
    print('train with default setting')
    input_file = '../../data/titles-en-train.labeled'
    model_file = '03-train-input-txt.model'
    perceptron.train(input_file, model_file)
    input_file = '../../data/titles-en-test.word'
    output_file = 'my_answer.labeled'
    perceptron.predict_all(model_file, input_file, output_file)
    perceptron.reset_model()

    # 94.084307%
    print('train with feature engineering')
    train_file = '../../data/titles-en-train.labeled'
    tfidf_model_file = 'tfidf.model'
    df = pd.read_table(train_file, header=None)
    X1 = df.iloc[:, 1]
    y = df.iloc[:, 0]

    test_file = '../../data/titles-en-test.word'
    X2 = pd.read_table(test_file, header=None).iloc[:, 0]

    X = pd.concat([X1, X2]).reset_index(drop=True)

    word_vectorizer = StemmedTfidVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english',
        ngram_range=(1, 3),
        max_features=30000)
    word_vectorizer.fit(X)
    X_vecs_word = word_vectorizer.transform(X1)

    char_vectorizer = StemmedTfidVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        stop_words='english',
        ngram_range=(2, 4),
        max_features=35000)
    char_vectorizer.fit(X)
    X_vecs_char = char_vectorizer.transform(X1)

    X_vecs = hstack([X_vecs_word, X_vecs_char])

    X_train, X_test, y_train, y_test = train_test_split(X_vecs, y, test_size=0.3, random_state=72)
    ppn = ppn(n_jobs=-1, random_state=72)
    ppn.fit(X_train, y_train)
    y_pred = ppn.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    print('誤分類: {}'.format((y_test != y_pred).sum()))
    print('正解率: %.5f' % acc)

    model_file = 'ppn_{}.pkl'.format(acc)
    with open(model_file, 'wb') as f:
        pickle.dump(ppn, f)

    print('test with test_file')
    X_vecs_word = word_vectorizer.transform(X2)
    X_vecs_char = char_vectorizer.transform(X2)
    X_vecs = hstack([X_vecs_word, X_vecs_char])

    y_pred = ppn.predict(X_vecs)
    ret = pd.concat([pd.Series(y_pred), X2], axis=1, ignore_index=True)
    ret.to_csv('my_answer_sklearn.labeled', sep='\t', encoding='utf-8', index=False, header=False)
