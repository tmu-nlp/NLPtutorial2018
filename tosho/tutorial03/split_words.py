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
                line = line.strip()
                splited = model.split_sentence(line)
                o.write(' '.join(splited) + '\n')