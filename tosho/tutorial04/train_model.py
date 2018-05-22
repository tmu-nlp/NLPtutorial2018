
if __name__ == '__main__':
    import os, sys
    sys.path.append(os.path.pardir)
    from common.pos_model import load_data, PosModel
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train-file', required=True)
    parser.add_argument('-m', '--model-file', required=True)
    arg = parser.parse_args()

    data = load_data(arg.train_file)

    model = PosModel()
    model.train(data)

    model.save_params(arg.model_file)

    print(f'model file saved to {arg.model_file}')