
if __name__ == '__main__':
    import os, sys
    sys.path.append(os.path.pardir)
    from common.pos_model import load_data, PosModel
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test-file', required=True)
    parser.add_argument('-m', '--model-file', required=True)
    arg = parser.parse_args()

    model = PosModel()
    model.load_params(arg.model_file)

    t_data =load_data(arg.test_file, mode='test')

    estimate = model.predict_pos(t_data)

    for line in estimate:
        print(' '.join(line))