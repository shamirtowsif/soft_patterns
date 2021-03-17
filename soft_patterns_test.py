#!/usr/bin/env python3
"""
Script to evaluate the accuracy of a model.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
from soft_patterns import evaluate_accuracy, SoftPatternClassifier, Semiring
from torch import FloatTensor, LongTensor, cat, mm, norm, randn, zeros, ones
from torch.nn.functional import sigmoid, log_softmax
from util import shuffled_chunked_sorted, identity, chunked_sorted, to_cuda, right_pad
from baselines.cnn import PooledCnnClassifier, max_pool_seq, cnn_arg_parser
from baselines.dan import DanClassifier
from baselines.lstm import AveragingRnnClassifier
import sys
import torch
import numpy as np
from torch.nn import LSTM
from data import vocab_from_text, read_embeddings, read_docs, read_labels
from rnn import Rnn

SCORE_IDX = 0
START_IDX_IDX = 1
END_IDX_IDX = 2


# TODO: refactor duplicate code with soft_patterns.py
def main():
    n = None
    mlp_hidden_dim = 25
    num_mlp_layers = 2

    validation_data_file = "./soft_patterns/data/test.data"
    dev_vocab = vocab_from_text(validation_data_file)
    print("Dev vocab size:", len(dev_vocab))

    embedding_file = "./soft_patterns/glove.6B.50d.txt"
    vocab, embeddings, word_dim = read_embeddings(embedding_file, dev_vocab)

    seed = 100
    torch.manual_seed(seed)
    np.random.seed(seed)

    patterns = "5-50_4-50_3-50_2-50"
    pattern_specs = OrderedDict(sorted(([int(y) for y in x.split("-")] for x in patterns.split("_")), key=lambda t: t[0]))
    num_padding_tokens = max(list(pattern_specs.keys())) - 1

    dev_input, _ = read_docs(validation_data_file, vocab, num_padding_tokens=num_padding_tokens)
    validation_label_file = "./soft_patterns/data/test.labels"
    dev_labels = read_labels(validation_label_file)
    dev_data = list(zip(dev_input, dev_labels))

    num_classes = len(set(dev_labels))
    print("num_classes:", num_classes)

    semiring = Semiring(zeros, ones, torch.add, torch.mul, sigmoid, identity)

    rnn = None

    model = SoftPatternClassifier(pattern_specs, mlp_hidden_dim, num_mlp_layers, num_classes, embeddings, vocab, semiring, 0.1, False, rnn, None, False, 0, False, None, None)

    input_model = "./soft_patterns/output/model_9.pth"
    state_dict = torch.load(input_model, map_location=lambda storage, loc: storage)

    model.load_state_dict(state_dict)

    test_acc = evaluate_accuracy(model, dev_data, 1, False)

    print("Test accuracy: {:>8,.3f}%".format(100*test_acc))

    return 0


if __name__ == '__main__':
    main()
