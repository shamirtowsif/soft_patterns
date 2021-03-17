#!/usr/bin/env python3 -u
"""
A text classification model that feeds the document scores from a bunch of
soft patterns into an MLP.
"""

import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
from time import monotonic
import numpy as np
import os
from tensorboardX import SummaryWriter
import torch
from torch import FloatTensor, LongTensor, cat, mm, norm, randn, zeros, ones
from torch.autograd import Variable
from torch.nn import Module, Parameter, NLLLoss, LSTM
from torch.nn.functional import sigmoid, log_softmax
from torch.nn.utils.rnn import pad_packed_sequence
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rnn import lstm_arg_parser, Rnn
from data import read_embeddings, read_docs, read_labels, vocab_from_text, Vocab, UNK_IDX, START_TOKEN_IDX, END_TOKEN_IDX
from mlp import MLP, mlp_arg_parser
from util import shuffled_chunked_sorted, identity, chunked_sorted, to_cuda, right_pad

CW_TOKEN = "CW"
EPSILON = 1e-10


def fixed_var(tensor):
    return Variable(tensor, requires_grad=False)


def argmax(output):
    """ only works for kxn tensors """
    _, am = torch.max(output, 1)
    return am


def normalize(data):
    length = data.size()[0]
    for i in range(length):
        data[i] = data[i] / norm(data[i])  # unit length


class Semiring:
    def __init__(self, zero, one, plus, times, from_float, to_float):
        self.zero = zero
        self.one = one
        self.plus = plus
        self.times = times
        self.from_float = from_float
        self.to_float = to_float


def neg_infinity(*sizes):
    return -100 * ones(*sizes)  # not really -inf, shh

SHARED_SL_PARAM_PER_STATE_PER_PATTERN = 1
SHARED_SL_SINGLE_PARAM = 2

### Adapted from AllenNLP
def enable_gradient_clipping(model, clip) -> None:
    if clip is not None and clip > 0:
        # Pylint is unable to tell that we're in the case that _grad_clipping is not None...
        # pylint: disable=invalid-unary-operand-type
        clip_function = lambda grad: grad.clamp(-clip, clip)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.register_hook(clip_function)



class Batch:
    """
    A batch of documents.
    Handles truncating documents to `max_len`, looking up word embeddings,
    and padding so that all docs in the batch have the same length.
    Makes a smaller vocab and embeddings matrix, only including words that are in the batch.
    """
    def __init__(self, docs, embeddings, cuda, word_dropout=0, max_len=-1):
        # print(docs)
        mini_vocab = Vocab.from_docs(docs, default=UNK_IDX, start=START_TOKEN_IDX, end=END_TOKEN_IDX)
        # Limit maximum document length (for efficiency reasons).
        if max_len != -1:
            docs = [doc[:max_len] for doc in docs]
        doc_lens = [len(doc) for doc in docs]
        self.doc_lens = cuda(torch.LongTensor(doc_lens))
        self.max_doc_len = max(doc_lens)
        if word_dropout:
            # for each token, with probability `word_dropout`, replace word index with UNK_IDX.
            docs = [
                [UNK_IDX if np.random.rand() < word_dropout else x for x in doc]
                for doc in docs
            ]
        # pad docs so they all have the same length.
        # we pad with UNK, whose embedding is 0, so it doesn't mess up sums or averages.
        docs = [right_pad(mini_vocab.numberize(doc), self.max_doc_len, UNK_IDX) for doc in docs]
        self.docs = [cuda(fixed_var(torch.LongTensor(doc))) for doc in docs]
        local_embeddings = [embeddings[i] for i in mini_vocab.names]
        self.embeddings_matrix = cuda(fixed_var(FloatTensor(local_embeddings).t()))

    def size(self):
        return len(self.docs)


class SoftPatternClassifier(Module):
    """
    A text classification model that feeds the document scores from a bunch of
    soft patterns into an MLP
    """
    def __init__(self,
                 pattern_specs,
                 mlp_hidden_dim,
                 num_mlp_layers,
                 num_classes,
                 embeddings,
                 vocab,
                 semiring,
                 bias_scale_param,
                 gpu=False,
                 rnn=None,
                 pre_computed_patterns=None,
                 no_sl=False,
                 shared_sl=False,
                 no_eps=False,
                 eps_scale=None,
                 self_loop_scale=None):
        super(SoftPatternClassifier, self).__init__()
        self.semiring = semiring
        self.vocab = vocab
        self.embeddings = embeddings

        self.to_cuda = to_cuda(gpu)

        self.total_num_patterns = sum(pattern_specs.values())
        print(self.total_num_patterns, pattern_specs)
        self.rnn = rnn
        self.mlp = MLP(self.total_num_patterns, mlp_hidden_dim, num_mlp_layers, num_classes)

        if self.rnn is None:
            self.word_dim = len(embeddings[0])
        else:
            self.word_dim = self.rnn.num_directions * self.rnn.hidden_dim
        self.num_diags = 1  # self-loops and single-forward-steps
        self.no_sl = no_sl
        self.shared_sl = shared_sl

        self.pattern_specs = pattern_specs
        self.max_pattern_length = max(list(pattern_specs.keys()))

        self.no_eps = no_eps
        self.bias_scale_param = bias_scale_param

        # Shared parameters between main path and self loop.
        # 1 -- one parameter per state per pattern
        # 2 -- a single global parameter
        if self.shared_sl > 0:
            if self.shared_sl == SHARED_SL_PARAM_PER_STATE_PER_PATTERN:
                shared_sl_data = randn(self.total_num_patterns, self.max_pattern_length)
            elif self.shared_sl == SHARED_SL_SINGLE_PARAM:
                shared_sl_data = randn(1)

            self.self_loop_scale = Parameter(shared_sl_data)
        elif not self.no_sl:
            if self_loop_scale is not None:
                self.self_loop_scale = self.semiring.from_float(self.to_cuda(fixed_var(FloatTensor([self_loop_scale]))))
            else:
                self.self_loop_scale = self.to_cuda(fixed_var(semiring.one(1)))
            self.num_diags = 2

        # end state index for each pattern
        end_states = [
            [end]
            for pattern_len, num_patterns in self.pattern_specs.items()
            for end in num_patterns * [pattern_len - 1]
        ]

        self.end_states = self.to_cuda(fixed_var(LongTensor(end_states)))

        diag_data_size = self.total_num_patterns * self.num_diags * self.max_pattern_length
        diag_data = randn(diag_data_size, self.word_dim)
        bias_data = randn(diag_data_size, 1)

        normalize(diag_data)

        if pre_computed_patterns is not None:
            diag_data, bias_data = self.load_pre_computed_patterns(pre_computed_patterns, diag_data, bias_data, pattern_specs)

        self.diags = Parameter(diag_data)

        # Bias term
        self.bias = Parameter(bias_data)

        if not self.no_eps:
            self.epsilon = Parameter(randn(self.total_num_patterns, self.max_pattern_length - 1))

        # TODO: learned? hyperparameter?
            # since these are currently fixed to `semiring.one`, they are not doing anything.
            if eps_scale is not None:
                self.epsilon_scale = self.semiring.from_float(self.to_cuda(fixed_var(FloatTensor([eps_scale]))))
            else:
                self.epsilon_scale = self.to_cuda(fixed_var(semiring.one(1)))

        print("# params:", sum(p.nelement() for p in self.parameters()))

    def get_transition_matrices(self, batch, dropout=None):
        b = batch.size()
        n = batch.max_doc_len
        if self.rnn is None:
            transition_scores = \
                self.semiring.from_float(mm(self.diags, batch.embeddings_matrix) + self.bias_scale_param * self.bias).t()
            if dropout is not None and dropout:
                transition_scores = dropout(transition_scores)
            batched_transition_scores = [
                torch.index_select(transition_scores, 0, doc) for doc in batch.docs
            ]
            batched_transition_scores = torch.cat(batched_transition_scores).view(
                b, n, self.total_num_patterns, self.num_diags, self.max_pattern_length)

        else:
            # run an RNN to get the word vectors to input into our soft-patterns
            outs = self.rnn.forward(batch, dropout=dropout)
            padded, _ = pad_packed_sequence(outs, batch_first=True)
            padded = padded.contiguous().view(b * n, self.word_dim).t()

            if dropout is not None and dropout:
                padded = dropout(padded)

            batched_transition_scores = \
                self.semiring.from_float(mm(self.diags, padded) + self.bias_scale_param * self.bias).t()

            if dropout is not None and dropout:
                batched_transition_scores = dropout(batched_transition_scores)

            batched_transition_scores = \
                batched_transition_scores.contiguous().view(
                    b,
                    n,
                    self.total_num_patterns,
                    self.num_diags,
                    self.max_pattern_length
                )
        # transition matrix for each token idx
        transition_matrices = [
            batched_transition_scores[:, word_index, :, :, :]
            for word_index in range(n)
        ]
        return transition_matrices

    def load_pre_computed_patterns(self, pre_computed_patterns, diag_data, bias_data, pattern_spec):
        """Loading a set of pre-coputed patterns into diagonal and bias arrays"""
        pattern_indices = dict((p,0) for p in pattern_spec)

        # First,view diag_data and bias_data as 4/3d tensors
        diag_data_size = diag_data.size()[0]
        diag_data = diag_data.view(self.total_num_patterns, self.num_diags, self.max_pattern_length, self.word_dim)
        bias_data = bias_data.view(self.total_num_patterns, self.num_diags, self.max_pattern_length)

        n = 0

        # Pattern indices: which patterns are we loading?
        # the pattern index from which we start loading each pattern length.
        for (i, patt_len) in enumerate(pattern_spec.keys()):
            pattern_indices[patt_len] = n
            n += pattern_spec[patt_len]

        # Loading all pre-computed patterns
        for p in pre_computed_patterns:
            patt_len = len(p) + 1

            # Getting pattern index in diagonal data
            index = pattern_indices[patt_len]

            # Loading diagonal and bias for p
            diag, bias = self.load_pattern(p)

            # Updating diagonal and bias
            diag_data[index, 1, :(patt_len-1), :] = diag
            bias_data[index, 1, :(patt_len-1)] = bias

            # Updating pattern_indices
            pattern_indices[patt_len] += 1

        return diag_data.view(diag_data_size, self.word_dim), bias_data.view(diag_data_size, 1)

    def load_pattern(self, patt):
        """Loading diagonal and bias of one pattern"""
        diag = EPSILON * torch.randn(len(patt), self.word_dim)
        bias = torch.zeros(len(patt))

        factor = 10

        # Traversing elements of pattern.
        for (i, element) in enumerate(patt):
            # CW: high bias (we don't care about the identity of the token
            if element == CW_TOKEN:
                bias[i] = factor
            else:
                # Concrete word: we do care about the token (low bias).
                bias[i] = -factor

                # If we have a word vector for this element, the diagonal value if this vector
                if element in self.vocab:
                    diag[i] = FloatTensor(factor*self.embeddings[self.vocab.index[element]])

        return diag, bias

    def forward(self, batch, debug=0, dropout=None):
        """ Calculate scores for one batch of documents. """
        time1 = monotonic()
        transition_matrices = self.get_transition_matrices(batch, dropout)
        time2 = monotonic()

        self_loop_scale = None

        if self.shared_sl:
            self_loop_scale = self.semiring.from_float(self.self_loop_scale)
        elif not self.no_sl:
            self_loop_scale = self.self_loop_scale

        batch_size = batch.size()
        num_patterns = self.total_num_patterns
        scores = self.to_cuda(fixed_var(self.semiring.zero(batch_size, num_patterns)))

        # to add start state for each word in the document.
        restart_padding = self.to_cuda(fixed_var(self.semiring.one(batch_size, num_patterns, 1)))

        zero_padding = self.to_cuda(fixed_var(self.semiring.zero(batch_size, num_patterns, 1)))

        eps_value = self.get_eps_value()

        batch_end_state_idxs = self.end_states.expand(batch_size, num_patterns, 1)
        hiddens = self.to_cuda(Variable(self.semiring.zero(batch_size,
                                                           num_patterns,
                                                           self.max_pattern_length)))
        # set start state (0) to 1 for each pattern in each doc
        # print("executed")
        # exit(0)
        hiddens[:, :, 0] = self.to_cuda(self.semiring.one(num_patterns, batch_size, 1)).squeeze()
        if debug % 4 == 3:
            all_hiddens = [hiddens[0, :, :]]
        for i, transition_matrix in enumerate(transition_matrices):
            hiddens = self.transition_once(eps_value,
                                           hiddens,
                                           transition_matrix,
                                           zero_padding,
                                           restart_padding,
                                           self_loop_scale)
            if debug % 4 == 3:
                all_hiddens.append(hiddens[0, :, :])

            # Look at the end state for each pattern, and "add" it into score
            end_state_vals = torch.gather(hiddens, 2, batch_end_state_idxs).view(batch_size, num_patterns)
            # but only update score when we're not already past the end of the doc
            active_doc_idxs = torch.nonzero(torch.gt(batch.doc_lens, i)).squeeze()
            scores[active_doc_idxs] = \
                self.semiring.plus(
                    scores[active_doc_idxs],
                    end_state_vals[active_doc_idxs]
                )

        if debug:
            time3 = monotonic()
            print("MM: {}, other: {}".format(round(time2 - time1, 3), round(time3 - time2, 3)))

        scores = self.semiring.to_float(scores)

        if debug % 4 == 3:
            return self.mlp.forward(scores), transition_matrices, all_hiddens
        elif debug % 4 == 1:
            return self.mlp.forward(scores), scores
        else:
            return self.mlp.forward(scores)

    def get_eps_value(self):
        return None if self.no_eps else self.semiring.times(
            self.epsilon_scale,
            self.semiring.from_float(self.epsilon)
        )

    def transition_once(self,
                        eps_value,
                        hiddens,
                        transition_matrix_val,
                        zero_padding,
                        restart_padding,
                        self_loop_scale):
        # Adding epsilon transitions (don't consume a token, move forward one state)
        # We do this before self-loops and single-steps.
        # We only allow zero or one epsilon transition in a row.
        if self.no_eps:
            after_epsilons = hiddens
        else:
            after_epsilons = \
                self.semiring.plus(
                    hiddens,
                    cat((zero_padding,
                         self.semiring.times(
                             hiddens[:, :, :-1],
                             eps_value  # doesn't depend on token, just state
                         )), 2)
                )

        after_main_paths = \
            cat((restart_padding,  # <- Adding the start state
                 self.semiring.times(
                     after_epsilons[:, :, :-1],
                     transition_matrix_val[:, :, -1, :-1])
                 ), 2)

        if self.no_sl:
            return after_main_paths
        else:
            self_loop_scale = self_loop_scale.expand(transition_matrix_val[:, :, 0, :].size()) \
                if self.shared_sl == SHARED_SL_PARAM_PER_STATE_PER_PATTERN else self_loop_scale

            # Adding self loops (consume a token, stay in same state)
            after_self_loops = self.semiring.times(
                self_loop_scale,
                self.semiring.times(
                    after_epsilons,
                    transition_matrix_val[:, :, 0, :]
                )
            )
            # either happy or self-loop, not both
            return self.semiring.plus(after_main_paths, after_self_loops)

    def predict(self, batch, debug=0):
        output = self.forward(batch, debug).data
        return [int(x) for x in argmax(output)]


def train_batch(model, batch, num_classes, gold_output, optimizer, loss_function, gpu=False, debug=0, dropout=None):
    """Train on one doc. """
    optimizer.zero_grad()
    time0 = monotonic()
    loss = compute_loss(model, batch, num_classes, gold_output, loss_function, gpu, debug, dropout)
    time1 = monotonic()
    loss.backward()
    time2 = monotonic()

    optimizer.step()
    if debug:
        time3 = monotonic()
        print("Time in loss: {}, time in backward: {}, time in step: {}".format(round(time1 - time0, 3),
                                                                                round(time2 - time1, 3),
                                                                                round(time3 - time2, 3)))
    return loss.data


def compute_loss(model, batch, num_classes, gold_output, loss_function, gpu, debug=0, dropout=None):
    time1 = monotonic()
    output = model.forward(batch, debug, dropout)

    if debug:
        time2 = monotonic()
        print("Forward total in loss: {}".format(round(time2 - time1, 3)))

    return loss_function(
        log_softmax(output).view(batch.size(), num_classes),
        to_cuda(gpu)(fixed_var(LongTensor(gold_output)))
    )


def evaluate_accuracy(model, data, batch_size, gpu, debug=0):
    n = float(len(data))
    correct = 0
    num_1s = 0
    for batch in chunked_sorted(data, batch_size):
        batch_obj = Batch([x for x, y in batch], model.embeddings, to_cuda(gpu))
        gold = [y for x, y in batch]
        predicted = model.predict(batch_obj, debug)
        num_1s += predicted.count(1)
        correct += sum(1 for pred, gold in zip(predicted, gold) if pred == gold)

    print("num predicted 1s:", num_1s)
    print("num gold 1s:     ", sum(gold == 1 for _, gold in data))

    return correct / n


def train(train_data,
          dev_data,
          model,
          num_classes,
          model_save_dir,
          num_iterations,
          model_file_prefix,
          learning_rate,
          batch_size,
          run_scheduler=False,
          gpu=False,
          clip=None,
          max_len=-1,
          debug=0,
          dropout=0,
          word_dropout=0,
          patience=1000):
    """ Train a model on all the given docs """

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_function = NLLLoss(None, False)

    enable_gradient_clipping(model, clip)

    if dropout:
        dropout = torch.nn.Dropout(dropout)
    else:
        dropout = None

    debug_print = int(100 / batch_size) + 1

    writer = None

    if model_save_dir is not None:
        writer = SummaryWriter(os.path.join(model_save_dir, "logs"))

    if run_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'min', 0.1, 10, True)

    best_dev_loss = 100000000
    best_dev_loss_index = -1
    best_dev_acc = -1
    start_time = monotonic()

    for it in range(num_iterations):
        np.random.shuffle(train_data)

        loss = 0.0
        i = 0
        for batch in shuffled_chunked_sorted(train_data, batch_size):
            batch_obj = Batch([x[0] for x in batch], model.embeddings, to_cuda(gpu), word_dropout, max_len)
            gold = [x[1] for x in batch]
            loss += torch.sum(
                train_batch(model, batch_obj, num_classes, gold, optimizer, loss_function, gpu, debug, dropout)
            )

            if i % debug_print == (debug_print - 1):
                print(".", end="", flush=True)
            i += 1

        if writer is not None:
            for name, param in model.named_parameters():
                writer.add_scalar("parameter_mean/" + name,
                                  param.data.mean(),
                                  it)
                writer.add_scalar("parameter_std/" + name, param.data.std(), it)
                if param.grad is not None:
                    writer.add_scalar("gradient_mean/" + name,
                                      param.grad.data.mean(),
                                      it)
                    writer.add_scalar("gradient_std/" + name,
                                      param.grad.data.std(),
                                      it)

            writer.add_scalar("loss/loss_train", loss, it)

        dev_loss = 0.0
        i = 0
        for batch in chunked_sorted(dev_data, batch_size):
            batch_obj = Batch([x[0] for x in batch], model.embeddings, to_cuda(gpu))
            gold = [x[1] for x in batch]
            dev_loss += torch.sum(compute_loss(model, batch_obj, num_classes, gold, loss_function, gpu, debug).data)

            if i % debug_print == (debug_print - 1):
                print(".", end="", flush=True)

            i += 1

        if writer is not None:
            writer.add_scalar("loss/loss_dev", dev_loss, it)
        print("\n")

        finish_iter_time = monotonic()
        train_acc = evaluate_accuracy(model, train_data[:1000], batch_size, gpu)
        dev_acc = evaluate_accuracy(model, dev_data, batch_size, gpu)

        print(
            "iteration: {:>7,} train time: {:>9,.3f}m, eval time: {:>9,.3f}m "
            "train loss: {:>12,.3f} train_acc: {:>8,.3f}% "
            "dev loss: {:>12,.3f} dev_acc: {:>8,.3f}%".format(
                it,
                (finish_iter_time - start_time) / 60,
                (monotonic() - finish_iter_time) / 60,
                loss / len(train_data),
                train_acc * 100,
                dev_loss / len(dev_data),
                dev_acc * 100
            )
        )

        if dev_loss < best_dev_loss:
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                print("New best acc!")
            print("New best dev!")
            best_dev_loss = dev_loss
            best_dev_loss_index = 0
            if model_save_dir is not None:
                model_save_file = os.path.join(model_save_dir, "{}_{}.pth".format(model_file_prefix, it))
                print("saving model to", model_save_file)
                torch.save(model.state_dict(), model_save_file)
        else:
            best_dev_loss_index += 1
            if best_dev_loss_index == patience:
                print("Reached", patience, "iterations without improving dev loss. Breaking")
                break

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            print("New best acc!")
            if model_save_dir is not None:
                model_save_file = os.path.join(model_save_dir, "{}_{}.pth".format(model_file_prefix, it))
                print("saving model to", model_save_file)
                torch.save(model.state_dict(), model_save_file)

        if run_scheduler:
            scheduler.step(dev_loss)

    return model


def main():
    patterns = "5-50_4-50_3-50_2-50"
    pattern_specs = OrderedDict(sorted(([int(y) for y in x.split("-")] for x in patterns.split("_")), key=lambda t: t[0]))

    pre_computed_patterns = None
    n = None
    mlp_hidden_dim = 25
    num_mlp_layers = 2

    seed = 100
    #Sets the seed for generating random numbers.
    torch.manual_seed(seed)
    #This method is called when RandomState is initialized.
    np.random.seed(seed)

    validation_data_file = "./soft_patterns/data/dev.data"
    dev_vocab = vocab_from_text(validation_data_file)
    # print(dev_vocab.index)
    print("Dev vocab size:", len(dev_vocab))
    # exit(0)
    train_data_file = "./soft_patterns/data/train.data"
    train_vocab = vocab_from_text(train_data_file)
    print("Train vocab size:", len(train_vocab))
    dev_vocab |= train_vocab


    embedding_file='./soft_patterns/glove.6B.50d.txt'
    vocab, embeddings, word_dim = read_embeddings(embedding_file, dev_vocab)

    num_padding_tokens = max(list(pattern_specs.keys())) - 1
    # print(num_padding_tokens)
    # exit(0)

    dev_input, _ = read_docs(validation_data_file, vocab, num_padding_tokens=num_padding_tokens)
    validation_label_file = "./soft_patterns/data/dev.labels"
    dev_labels = read_labels(validation_label_file)
    dev_data = list(zip(dev_input, dev_labels))

    # print(dev_data[50][0])
    # print(len(dev_data[50][0]))
    # exit(0)

    np.random.shuffle(dev_data)
    num_iterations = 10

    train_input, _ = read_docs(train_data_file, vocab, num_padding_tokens=num_padding_tokens)
    train_labels_file = "./soft_patterns/data/train.labels"
    train_labels = read_labels(train_labels_file)

    print("training instances:", len(train_input))

    num_classes = len(set(train_labels))

    train_data = list(zip(train_input, train_labels))
    np.random.shuffle(train_data)

    print("num_classes:", num_classes)
    rnn = None
    semiring = Semiring(zeros, ones, torch.add, torch.mul, sigmoid, identity)

    model = SoftPatternClassifier(pattern_specs, mlp_hidden_dim, num_mlp_layers, num_classes, embeddings, vocab, semiring, 0.1, False, rnn, pre_computed_patterns, False, 0, False, None, None)

    model_file_prefix = "model"
    model_save_dir = "./soft_patterns/output/"

    print("Training with", model_file_prefix)

    train(train_data, dev_data, model, num_classes, model_save_dir, num_iterations, model_file_prefix, 0.001, 1, False, False, None, -1,0,0,0, 30)

    return 0

if __name__ == '__main__':
    main()
