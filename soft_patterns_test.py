#!/usr/bin/env python3
"""
Script to visualize the patterns in a SoftPatterns model based on their
highest-scoring spans in the dev set.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
from baselines.dan import DanClassifier
from baselines.lstm import AveragingRnnClassifier
import sys
import torch
import numpy as np
from torch.nn import LSTM
from data import vocab_from_text, read_embeddings, read_docs, read_labels
from soft_patterns import MaxPlusSemiring, LogSpaceMaxTimesSemiring, evaluate_accuracy, SoftPatternClassifier, ProbSemiring, training_arg_parser, \
    soft_pattern_arg_parser
from rnn import lstm_arg_parser, Rnn

SCORE_IDX = 0
START_IDX_IDX = 1
END_IDX_IDX = 2


# TODO: refactor duplicate code with soft_patterns.py
def main(args):
    print(args)

    n = args.num_train_instances
    mlp_hidden_dim = args.mlp_hidden_dim
    num_mlp_layers = args.num_mlp_layers

    dev_vocab = vocab_from_text(args.vd)
    print("Dev vocab size:", len(dev_vocab))

    vocab, embeddings, word_dim = \
        read_embeddings(args.embedding_file, dev_vocab)

    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    dev_input, dev_text = read_docs(args.vd, vocab, 0)
    dev_labels = read_labels(args.vl)
    dev_data = list(zip(dev_input, dev_labels))
    if n is not None:
        dev_data = dev_data[:n]

    num_classes = len(set(dev_labels))
    print("num_classes:", num_classes)

    if args.dan:
        model = DanClassifier(mlp_hidden_dim,
                                  num_mlp_layers,
                                  num_classes,
                                  embeddings,
                                  args.gpu)
    elif args.bilstm:
        cell_type = LSTM

        model = AveragingRnnClassifier(args.hidden_dim,
                                       mlp_hidden_dim,
                                       num_mlp_layers,
                                       num_classes,
                                       embeddings,
                                       cell_type=cell_type,
                                       gpu=args.gpu)
    else:
        pattern_specs = OrderedDict(sorted(([int(y) for y in x.split("-")] for x in args.patterns.split("_")),
                                           key=lambda t: t[0]))

        semiring = \
            MaxPlusSemiring if args.maxplus else (
                LogSpaceMaxTimesSemiring if args.maxtimes else ProbSemiring
            )

        if args.use_rnn:
            rnn = Rnn(word_dim,
                  args.hidden_dim,
                  cell_type=LSTM,
                  gpu=args.gpu)
        else:
            rnn = None

        model = SoftPatternClassifier(pattern_specs, mlp_hidden_dim, num_mlp_layers, num_classes, embeddings, vocab,
                                      semiring, args.gpu, rnn)

    if args.gpu:
        model.to_cuda(model)

    # Loading model
    state_dict = torch.load(args.input_model)
    model.load_state_dict(state_dict)

    test_acc = evaluate_accuracy(model, dev_data, args.batch_size, args.gpu)

    print("Test accuray: {:>8,.3f}%".format(100*test_acc))

    return 0


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            parents=[soft_pattern_arg_parser(), training_arg_parser()])
    parser.add_argument("--dan", help="Dan classifier", action='store_true')
    parser.add_argument("--bilstm", help="BiLSTM classifier", action='store_true')

    sys.exit(main(parser.parse_args()))
