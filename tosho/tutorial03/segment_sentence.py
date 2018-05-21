import os, sys
sys.path.append(os.path.pardir)
from common.n_gram import NGram
from common.smoothings import SimpleSmoothing

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test-file')
    parser.add_argument('-m', '--model-file')
    parser.add_argument('-o', '--output-file')
    arg = parser.parse_args()

    model = NGram(1)
    model.load_params(arg.model_file)

    smoothing = SimpleSmoothing()

    model.set_smoothing(smoothing)

    with open(arg.test_file, 'r') as f:
        with open(arg.output_file, 'w') as o:
            for line in f:
                line = line.strip('\n')
                splited = model.segment(line)
                o.write(' '.join(splited) + '\n')

'''
$ python segment_sentence.py -t ../../data/wiki-ja-test.txt -m wiki-ja.pkl -o wiki-ja-answer.word
$ ../../script/gradews.pl ../../data/wiki-ja-test.word wiki-ja-answer.word
Sent Accuracy: 23.81% (20/84)
Word Prec: 68.93% (1861/2700)
Word Rec: 80.77% (1861/2304)
F-meas: 74.38%
Bound Accuracy: 83.25% (2683/3223)
'''