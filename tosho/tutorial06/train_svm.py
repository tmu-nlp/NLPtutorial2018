# python train_perceptron.py

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.pardir)

    from common.svm import BinaryClassifier, SimpleOptimizer, Trainer, NormalizingOptimizer
    from common.utils import load_labeled_data, load_word_data
    import matplotlib.pyplot as plt

    model = BinaryClassifier()
    optimizer = SimpleOptimizer(lr=1)
    # optimizer = NormalizingOptimizer()

    train_data = list(load_labeled_data('../../data/titles-en-train.labeled'))
    # train_data = [('A site , located in Maizuru , Kyoto'.split(' '),-1)]
    
    print(f'train data: {len(train_data)}')

    trainer = Trainer(model, train_data, epochs=20, optimizer=optimizer)
    trainer.train()

    # for key, value in model.params.items():
    #     print(f'{key} : {value}')

    # model.save_params('model.pkl')


    # grad = model.gradient(train_data[0][0], train_data[0][1])
    # print('='*20)
    # for key, value in grad.items():
    #     print(f'{key} : {value}')
    # optimizer.update(model.params, grad)

    # train_data = [('Shoken , monk born in Kyoto'.split(' '), 1)]
    # grad = model.gradient(train_data[0][0], train_data[0][1])
    # print('='*20)
    # for key, value in grad.items():
    #     print(f'{key} : {value}')    
    # optimizer.update(model.params, grad)
    # print('='*20)
    # for key, value in model.params.items():
    #     print(f'{key} : {value}')