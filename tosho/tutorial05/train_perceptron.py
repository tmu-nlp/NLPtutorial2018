# python train_perceptron.py

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.pardir)

    from common.binary_classifier import BinaryClassifier, SimpleOptimizer, Trainer, NormalizingOptimizer
    from common.utils import load_labeled_data, load_word_data
    import matplotlib.pyplot as plt

    model = BinaryClassifier()
    optimizer = SimpleOptimizer()
    # optimizer = NormalizingOptimizer()

    train_data = list(load_labeled_data('../../data/titles-en-train.labeled'))
    
    print(f'train data: {len(train_data)}')

    trainer = Trainer(model, train_data, epochs=40, optimizer=optimizer, mini_batch_size=1000)
    trainer.train()

    trainer.draw_accuracy('model.png')

    model.save_params('model.pkl')
