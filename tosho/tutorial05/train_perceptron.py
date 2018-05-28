if __name__ == '__main__':
    import sys, os
    sys.path.append(os.pardir)

    from common.binary_classifier import BinaryClassifier, BinaryClassifierOptimizer, Trainer
    from common.utils import load_labeled_data, load_word_data
    import matplotlib.pyplot as plt

    model = BinaryClassifier()

    train_data = list(load_labeled_data('../../data/titles-en-train.labeled'))
    test_data = list(load_labeled_data('../../data/titles-en-test.labeled'))
    
    print(f'train data: {len(train_data)} | test data: {len(test_data)}')

    trainer = Trainer(model, train_data, test_data, epochs=50)
    trainer.train()

    model.save_params('model.pkl')

    plt.plot(trainer.train_acc_list, trainer.test_acc_list)
    fig = plt.gcf()
    plt.show()
    plt.draw()
    fig.savefig('naive_classifier.png', dpi=100)
