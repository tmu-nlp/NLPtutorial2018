import os, sys
sys.path.append(os.path.pardir)
from common.n_gram import NGram

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train-file')
    parser.add_argument('-o', '--output-file')
    arg = parser.parse_args()

    model = NGram(1)
    with open(arg.train_file, 'r') as f:
        model.train(list(f))
    
    model.print_params()

    model.save_params(arg.output_file)

    print(f'saved parameter file to {arg.output_file}')