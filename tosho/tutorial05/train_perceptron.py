if __name__ == '__main__':
    import sys, os
    sys.path.append(os.pardir)

    from common.binary_classifier import BinaryClassifier, SimpleOptimizer, Trainer
    from common.utils import load_labeled_data, load_word_data
    import matplotlib.pyplot as plt

    model = BinaryClassifier()
    optimizer = SimpleOptimizer()

    train_data = list(load_labeled_data('../../data/titles-en-train.labeled'))
    test_data = list(load_labeled_data('../../data/titles-en-test.labeled'))
    
    print(f'train data: {len(train_data)} | test data: {len(test_data)}')

    trainer = Trainer(model, train_data, test_data, epochs=100, optimizer=optimizer)
    trainer.train()

    trainer.draw_accuracy('model.png')

    model.save_params('model.pkl')
