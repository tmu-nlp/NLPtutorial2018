# python train_perceptron.py

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.pardir)

    from common.svm import BinaryClassifier, SimpleOptimizer, Trainer
    from common.utils import load_labeled_data, load_word_data
    import matplotlib.pyplot as plt

    model = BinaryClassifier()
    optimizer = SimpleOptimizer(lr=0.1, thres=0.0001)

    train_data = list(load_labeled_data('../../data/titles-en-train.labeled'))
    
    print(f'train data: {len(train_data)}')

    trainer = Trainer(model, train_data, epochs=40, optimizer=optimizer)
    trainer.train()

    model.save_params('model.pkl')