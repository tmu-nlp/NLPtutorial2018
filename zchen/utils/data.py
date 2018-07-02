import random

def load_train_dataset(featurizer, fname = '../../data/titles-en-train.labeled'):
    with open(fname) as fr:
        for line in fr:
            label, sentence = line.split('\t') # tokenize & feature extraction
            yield featurizer(sentence), int(label)

def split_dataset(inputs, outputs, train_set_ratio, shuffle = True):
    total = len(inputs)
    idx = list(range(total))
    if shuffle: random.shuffle(idx)
    train_set_ratio = int(total * train_set_ratio)
    train_set_idx = idx[:train_set_ratio]
    valid_set_idx = idx[train_set_ratio:]
    validation_x = tuple(inputs [i] for i in valid_set_idx)
    validation_y = tuple(outputs[i] for i in valid_set_idx)
    return total, train_set_idx, (validation_x, validation_y)
