from itertools import chain
from util import nub
import numpy as np
import string
from collections import OrderedDict

UNK_TOKEN = "*UNK*"
START_TOKEN = "*START*"
END_TOKEN = "*END*"
PRINTABLE = set(string.printable)

def main():
    validation_data_file, validation_label_file, train_data_file, train_label_file = "./soft_patterns/data/dev.data", "./soft_patterns/data/dev.labels", "./soft_patterns/data/train.data", "./soft_patterns/data/train.labels"

    dev_docs, dev_names, dev_index = [], [], {}
    with open(validation_data_file, encoding="ISO-8859-1") as input_file:
        for line in input_file: dev_docs.append(line.strip().split())
    for doc in dev_docs:
        for i in doc:
            dev_names.append(i)
    dev_names = list(nub(chain([UNK_TOKEN, START_TOKEN, END_TOKEN], dev_names)))

    for i, name in enumerate(dev_names): dev_index[name] = i

    train_docs, train_names, train_index = [], [], {}
    with open(train_data_file, encoding="ISO-8859-1") as input_file:
        for line in input_file: train_docs.append(line.strip().split())
    for doc in train_docs:
        for i in doc: train_names.append(i)
    train_names = list(nub(chain([UNK_TOKEN, START_TOKEN, END_TOKEN], train_names)))

    for i, name in enumerate(train_names): train_index[name] = i

    new_dev_names, new_dev_index = list(nub(chain([UNK_TOKEN, START_TOKEN, END_TOKEN], dev_names+train_names))), {}
    for i, name in enumerate(new_dev_names): new_dev_index[name] = i

    embedding_file = "./soft_patterns/glove.6B.50d.txt"
    dim = 50

    embedding_names, embedding_index, word_vecs = [], {}, []
    with open(embedding_file, encoding="utf-8") as input_file:
        for line in input_file:
            word, vec_str = line.strip().split(' ', 1)
            if all(c in PRINTABLE for c in word) and word in new_dev_names:
                word_vecs.append((word, np.fromstring(vec_str, dtype=float, sep=' ')))
                embedding_names.append(word)
    embedding_names = list(nub(chain([UNK_TOKEN, START_TOKEN, END_TOKEN], embedding_names)))
    for i, name in enumerate(embedding_names): embedding_index[name] = i
    embedding_vectors = [np.zeros(dim), np.zeros(dim), np.zeros(dim)] + [vec/np.linalg.norm(vec) for word, vec in word_vecs]

    patterns = "5-50_4-50_3-50_2-50"
    pattern_specs = OrderedDict(sorted(([int(x) for x in pattern.split('-')] for pattern in patterns.split('_')), key = lambda t: t[0]))

    num_padding_tokens = max(list(pattern_specs.keys())) - 1
    
    dev_docs = []
    with open(validation_data_file, encoding="ISO-8859-1") as input_file:
        for line in input_file:
            dev_docs.append(line.strip().split())

    dev_input = []
    for doc in dev_docs:
        dev_input.append(([START_TOKEN]*num_padding_tokens) + [embedding_index.get(token, UNK_TOKEN) for token in doc] + ([END_TOKEN]*num_padding_tokens))

    dev_labels = []
    with open(validation_label_file) as input_file:
        for line in input_file:
            dev_labels.append(int(line.strip()))

    dev_data = list(zip(dev_input, dev_labels))

    train_input = []
    for doc in train_docs:
        train_input.append(([START_TOKEN]*num_padding_tokens) + [embedding_index.get(token, UNK_TOKEN) for token in doc] + ([END_TOKEN]*num_padding_tokens))

    train_labels = []
    with open(train_label_file) as input_file:
        for line in input_file:
            train_labels.append(int(line.strip()))

    train_data = list(zip(train_input, train_labels))


if __name__ == "__main__":
    main()
