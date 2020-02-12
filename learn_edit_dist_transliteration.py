#!/usr/bin/env python3


import argparse
from itertools import chain
import math

import torch
from torch import nn, optim
from torch.functional import F
from torchtext import data


MINF = torch.log(torch.tensor(0.))


class Encoder(nn.Module):
    def __init__(self, vocab, n_layers, hidden_dim, n_heads):
        super(Encoder, self).__init__()

        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(len(vocab), hidden_dim)
        self.pe = PositionalEncoding(hidden_dim)
        self.layers = nn.ModuleList(
            nn.modules.TransformerEncoderLayer(
                hidden_dim, n_heads, 2 * hidden_dim) for _ in range(n_layers))

    def forward(self, data):
        output = self.pe(self.embeddings(data) * math.sqrt(self.hidden_dim))
        mask = (data != 1)

        for layer in self.layers:
            output = layer(output)#, src_key_padding_mask=mask.transpose(1, 0))

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EditDistBase(nn.Module):
    def __init__(self, ar_vocab, en_vocab):
        super(EditDistBase, self).__init__()

        self.ar_symbol_count = len(ar_vocab)
        self.en_symbol_count = len(en_vocab)

        self.n_target_classes = (
            1 + # special termination symbol
            self.ar_symbol_count + # delete arabic
            self.en_symbol_count + # insert english
            self.ar_symbol_count * self.en_symbol_count) # substitute

    def _deletion_id(self, ar_char):
        return 1 + ar_char

    def _insertion_id(self, en_char):
        return 1 + self.ar_symbol_count + en_char

    def _substitute_id(self, ar_char, en_char):
        subs_id = (1 + self.ar_symbol_count + self.en_symbol_count +
                self.en_symbol_count * ar_char + en_char)
        assert subs_id < self.n_target_classes
        return subs_id


class EditDistNeuralModel(EditDistBase):
    def __init__(self, ar_vocab, en_vocab):
        super(EditDistNeuralModel, self).__init__(ar_vocab, en_vocab)

        self.ar_encoder = Encoder(ar_vocab, 1, 64, 4)
        self.en_encoder = Encoder(en_vocab, 1, 64, 4)

        self.scorer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(2 * 64, 64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(64, self.n_target_classes))


    def _forward_evaluation(self, ar_sent, en_sent, action_scores):
        plausible_deletions = torch.zeros_like(action_scores) + MINF
        plausible_insertions = torch.zeros_like(action_scores) + MINF
        plausible_substitutions = torch.zeros_like(action_scores) + MINF

        alpha = []
        for t, ar_char in enumerate(ar_sent[:, 0]):
            alpha.append([])
            for v, en_char in enumerate(en_sent[:, 0]):
                if t == 0 and v == 0:
                    alpha[0].append(torch.tensor(0.))
                    continue

                deletion_id = self._deletion_id(ar_char)
                insertion_id = self._insertion_id(en_char)
                subsitute_id = self._substitute_id(ar_char, en_char)

                plausible_deletions[t, v, deletion_id] = 0
                plausible_insertions[t, v, insertion_id] = 0
                plausible_substitutions[t, v, subsitute_id] = 0

                to_sum = []
                if v >= 1:
                    to_sum.append(action_scores[t, v, insertion_id] + alpha[t][v - 1])
                if t >= 1:
                    to_sum.append(action_scores[t, v, deletion_id] + alpha[t - 1][v])
                if v >= 1 and t >= 1:
                    to_sum.append(action_scores[t, v, subsitute_id] + alpha[t - 1][v - 1])

                if not to_sum:
                    alpha[t].append(MINF)
                if len(to_sum) == 1:
                    alpha[t].append(to_sum[0])
                else:
                    alpha[t].append(torch.stack(to_sum).logsumexp(0))

        alpha_tensor = torch.stack([torch.stack(v) for v in alpha])
        return (alpha_tensor, plausible_deletions, plausible_insertions, plausible_substitutions)

    def forward(self, ar_sent, en_sent):
        ar_len, en_len = ar_sent.size(0), en_sent.size(0)
        ar_vectors = self.ar_encoder(ar_sent)
        en_vectors = self.en_encoder(en_sent)

        feature_table = torch.cat((
            ar_vectors.repeat(1, en_len, 1),
            en_vectors.squeeze(1).unsqueeze(0).repeat(ar_len, 1, 1)), dim=2)
        action_scores = F.log_softmax(self.scorer(feature_table), dim=2)
        action_entropy = -(action_scores * action_scores.exp()).sum()

        alpha, plausible_deletions, plausible_insertions, plausible_substitutions = self._forward_evaluation(
                ar_sent, en_sent, action_scores)

        with torch.no_grad():
            beta = torch.zeros((ar_len, en_len)) + torch.log(torch.tensor(0.))
            beta[-1, -1] = 0.0

            for t, ar_char in reversed(list(enumerate(ar_sent[:, 0]))):
                for v, en_char in reversed(list(enumerate(en_sent[:, 0]))):
                    deletion_id = self._deletion_id(ar_char)
                    insertion_id = self._insertion_id(en_char)
                    subsitute_id = self._substitute_id(ar_char, en_char)

                    to_sum = [beta[t, v]]
                    if v < en_len - 1:
                        to_sum.append(action_scores[t, v + 1, insertion_id] + beta[t, v + 1])
                    if t < ar_len - 1:
                        to_sum.append(
                            action_scores[t + 1, v, deletion_id] + beta[t + 1, v])
                    if v < en_len - 1 and t < ar_len - 1:
                        to_sum.append(
                            action_scores[t + 1, v + 1, subsitute_id] + beta[t + 1, v + 1])
                    beta[t, v] = torch.logsumexp(torch.tensor(to_sum), dim=0)

            # deletion expectation
            expected_deletions = torch.zeros_like(action_scores) + MINF
            expected_deletions[1:, :] = (
                alpha[:-1, :].unsqueeze(2) +
                action_scores[1:, :] + plausible_deletions[1:, :] +
                beta[1:, :].unsqueeze(2))
            # insertions expectation
            expected_insertions = torch.zeros_like(action_scores) + MINF
            expected_insertions[:, 1:] = (
                alpha[:, :-1].unsqueeze(2) +
                action_scores[:, 1:] + plausible_insertions[:, 1:] +
                beta[:, 1:].unsqueeze(2))
            # substitution expectation
            expected_substitutions = torch.zeros_like(action_scores) + MINF
            expected_substitutions[1:, 1:] = (
                alpha[:-1, :-1].unsqueeze(2) +
                action_scores[1:, 1:] + plausible_substitutions[1:, 1:] +
                beta[1:, 1:].unsqueeze(2))

            expected_counts = torch.stack([
                expected_deletions, expected_insertions, expected_substitutions]).logsumexp(0)
            expected_counts -= expected_counts.logsumexp(2, keepdim=True)

        return action_scores, torch.exp(expected_counts), action_entropy, alpha[-1, -1]

    def viterbi(self, ar_sent, en_sent):
        zero = torch.tensor(0)
        ar_len, en_len = ar_sent.size(0), en_sent.size(0)
        ar_vectors = self.ar_encoder(ar_sent)
        en_vectors = self.en_encoder(en_sent)

        feature_table = torch.cat((
            ar_vectors.repeat(1, en_len, 1),
            en_vectors.squeeze(1).unsqueeze(0).repeat(ar_len, 1, 1)), dim=2)
        actions = F.log_softmax(self.scorer(feature_table), dim=2)

        with torch.no_grad():
            action_count = torch.zeros((ar_len, en_len))
            alpha = torch.zeros((ar_len, en_len)) + MINF
            alpha[0, 0] = 0
            for t, ar_char in enumerate(ar_sent[:, 0]):
                for v, en_char in enumerate(en_sent[:, 0]):
                    if t == 0 and v == 0:
                        continue

                    deletion_id = self._deletion_id(ar_char)
                    insertion_id = self._insertion_id(en_char)
                    subsitute_id = self._substitute_id(ar_char, en_char)

                    possible_actions = []

                    if v >= 1:
                        possible_actions.append(
                            (actions[t, v, insertion_id] + alpha[t, v - 1],
                             action_count[t, v - 1] + 1))
                    if t >= 1:
                        possible_actions.append(
                            (actions[t, v, deletion_id] + alpha[t - 1, v],
                             action_count[t - 1, v] + 1))
                    if v >= 1 and t >= 1:
                        possible_actions.append(
                            (actions[t, v, subsitute_id] + alpha[t - 1, v - 1],
                             action_count[t - 1, v - 1] + 1))

                    best_action_cost, best_action_count = max(
                        possible_actions, key=lambda x: x[0] / x[1])

                    alpha[t, v] = best_action_cost
                    action_count[t, v] = best_action_count

        return torch.exp(alpha[-1, -1] / action_count[-1, -1])


class EditDistStatModel(EditDistBase):
    def __init__(self, ar_vocab, en_vocab):
        super(EditDistStatModel, self).__init__(ar_vocab, en_vocab)

        self.weights = torch.log(torch.tensor(
            [1 / self.n_target_classes for _ in range(self.n_target_classes)]))

    def forward(self, ar_sent, en_sent):
        ar_len, en_len = ar_sent.size(0), en_sent.size(0)
        table_shape = ((ar_len, en_len, self.n_target_classes))

        plausible_deletions = torch.zeros(table_shape) + MINF
        plausible_insertions = torch.zeros(table_shape) + MINF
        plausible_substitutions = torch.zeros(table_shape) + MINF

        alpha = torch.zeros((ar_len, en_len)) + MINF
        alpha[0, 0] = 0
        for t, ar_char in enumerate(ar_sent[:, 0]):
            for v, en_char in enumerate(en_sent[:, 0]):
                if t == 0 and v == 0:
                    continue

                deletion_id = self._deletion_id(ar_char)
                insertion_id = self._insertion_id(en_char)
                subsitute_id = self._substitute_id(ar_char, en_char)

                plausible_deletions[t, v, deletion_id] = 0
                plausible_insertions[t, v, insertion_id] = 0
                plausible_substitutions[t, v, subsitute_id] = 0

                to_sum = [alpha[t, v]]
                if v >= 1:
                    to_sum.append(self.weights[insertion_id] + alpha[t, v - 1])
                if t >= 1:
                    to_sum.append(self.weights[deletion_id] + alpha[t - 1, v])
                if v >= 1 and t >= 1:
                    to_sum.append(self.weights[subsitute_id] + alpha[t - 1, v - 1])

                alpha[t, v] = torch.logsumexp(torch.tensor(to_sum), dim=0)

        beta = torch.zeros((ar_len, en_len)) + MINF
        beta[-1, -1] = 0.0

        for t, ar_char in reversed(list(enumerate(ar_sent[:, 0]))):
            for v, en_char in reversed(list(enumerate(en_sent[:, 0]))):
                deletion_id = self._deletion_id(ar_char)
                insertion_id = self._insertion_id(en_char)
                subsitute_id = self._substitute_id(ar_char, en_char)

                to_sum = [beta[t, v]]
                if v < en_len - 1:
                    to_sum.append(
                        self.weights[insertion_id] + beta[t, v + 1])
                if t < ar_len - 1:
                    to_sum.append(
                        self.weights[deletion_id] + beta[t + 1, v])
                if v < en_len - 1 and t < ar_len - 1:
                    to_sum.append(
                        self.weights[subsitute_id] + beta[t + 1, v + 1])
                beta[t, v] = torch.logsumexp(torch.tensor(to_sum), dim=0)

        expand_weights = self.weights.unsqueeze(0).unsqueeze(0)
        # deletion expectation
        expected_deletions = (
            alpha[:-1, :].unsqueeze(2) +
            expand_weights + plausible_deletions[1:, :] +
            beta[1:, :].unsqueeze(2))
        # insertoin expectation
        expected_insertions = (
            alpha[:, :-1].unsqueeze(2) +
            expand_weights + plausible_insertions[:, 1:] +
            beta[:, 1:].unsqueeze(2))
        # substitution expectation
        expected_substitutions = (
            alpha[:-1, :-1].unsqueeze(2) +
            expand_weights + plausible_substitutions[1:, 1:] +
            beta[1:, 1:].unsqueeze(2))

        all_counts = torch.cat([
            expected_deletions.reshape((-1, self.n_target_classes)),
            expected_insertions.reshape((-1, self.n_target_classes)),
            expected_substitutions.reshape((-1, self.n_target_classes))],
            dim=0)

        expected_counts = all_counts.logsumexp(0)
        return expected_counts

    def viterbi(self, ar_sent, en_sent):
        zero = torch.tensor(0)
        ar_len, en_len = ar_sent.size(0), en_sent.size(0)

        action_count = torch.zeros((ar_len, en_len))
        alpha = torch.zeros((ar_len, en_len)) - MINF
        alpha[0, 0] = 0
        for t, ar_char in enumerate(ar_sent[:, 0]):
            for v, en_char in enumerate(en_sent[:, 0]):
                if t == 0 and v == 0:
                    continue

                deletion_id = self._deletion_id(ar_char)
                insertion_id = self._insertion_id(en_char)
                subsitute_id = self._substitute_id(ar_char, en_char)

                possible_actions = []

                if v >= 1:
                    possible_actions.append(
                        (self.weights[insertion_id] + alpha[t, v - 1],
                         action_count[t, v - 1] + 1))
                if t >= 1:
                    possible_actions.append(
                        (self.weights[deletion_id] + alpha[t - 1, v],
                         action_count[t - 1, v] + 1))
                if v >= 1 and t >= 1:
                    possible_actions.append(
                        (self.weights[subsitute_id] + alpha[t - 1, v - 1],
                         action_count[t - 1, v - 1] + 1))

                best_action_cost, best_action_count = max(
                    possible_actions, key=lambda x: x[0] / x[1])

                alpha[t, v] = best_action_cost
                action_count[t, v] = best_action_count

        return torch.exp(alpha[-1, -1] / action_count[-1, -1])

    def maximize_expectation(self, expectations):
        epsilon = torch.log(torch.tensor(1e-16)) + torch.zeros_like(self.weights)
        expecation_sum = torch.stack([epsilon] + expectations).logsumexp(0)
        distribution = (
            expecation_sum - expecation_sum.logsumexp(0, keepdim=True))

        self.weights = torch.stack([
            torch.log(torch.tensor(0.9)) + self.weights,
            torch.log(torch.tensor(0.1)) + distribution]).logsumexp(0)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("data_prefix", type=str)
    args = parser.parse_args()

    ar_text_field = data.Field(tokenize=list, init_token="<s>", eos_token="</s>")
    en_text_field = data.Field(tokenize=list, init_token="<s>", eos_token="</s>")
    labels_field = data.Field(sequential=False)

    train_data, val_data, test_data = data.TabularDataset.splits(
        path=args.data_prefix, train='train.txt',
        validation='eval.txt', test='test.txt', format='tsv',
        fields=[('ar', ar_text_field), ('en', en_text_field),
                ('labels', labels_field)])

    ar_text_field.build_vocab(train_data)
    en_text_field.build_vocab(train_data)
    labels_field.build_vocab(train_data)
    true_class_label = labels_field.vocab.stoi['1']

    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train_data, val_data, test_data), batch_sizes=(1, 1, 1),
        shuffle=True, device=0, sort_key=lambda x: len(x.ar))

    neural_model = EditDistNeuralModel(ar_text_field.vocab, en_text_field.vocab)
    stat_model = EditDistStatModel(ar_text_field.vocab, en_text_field.vocab)

    loss_function = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(neural_model.parameters())

    en_examples = []
    ar_examples = []
    labels = []
    pos_examples = 0
    stat_expecations = []
    for i, train_ex in enumerate(train_iter):
        label = 1 if train_ex.labels[0] == true_class_label else -1

        if not label:
            continue
        pos_examples += 1

        action_scores, expected_counts, action_entropy, logprob = neural_model(train_ex.ar, train_ex.en)
        if label == 1:
            exp_counts = stat_model(train_ex.ar, train_ex.en)
            stat_expecations.append(exp_counts)

        #loss = loss_function(action_scores, expected_counts) + label * logprob
        loss = -label * logprob
        if label == 1:
            loss += loss_function(action_scores, expected_counts)
        loss.backward()

        if pos_examples % 50 == 49:
            print(f"train loss = {loss.cpu():.10g}")
            optimizer.step()
            optimizer.zero_grad()

            stat_model.maximize_expectation(stat_expecations)
            entropy = -(stat_model.weights * stat_model.weights.exp()).sum()
            print(f"stat. model entropy = {entropy.cpu():.10g}")
            stat_expecations = []

        if pos_examples % 50 == 49:
            neural_model.eval()
            with torch.no_grad():
                neural_false_scores = []
                neural_true_scores = []
                stat_false_scores = []
                stat_true_scores = []
                for j, val_ex in enumerate(val_iter):
                    neural_score = neural_model.viterbi(val_ex.ar, val_ex.en)
                    stat_score = stat_model.viterbi(val_ex.ar, val_ex.en)
                    if j < 50:
                        if val_ex.labels == true_class_label:
                            neural_true_scores.append(neural_score)
                            stat_true_scores.append(stat_score)
                        else:
                            neural_false_scores.append(neural_score)
                            stat_false_scores.append(stat_score)
                    else:
                        print(f"neural true  scores: {sum(neural_true_scores) / len(neural_true_scores)}")
                        print(f"neural false scores: {sum(neural_false_scores) / len(neural_false_scores)}")
                        print(f"stat true  scores:   {sum(stat_true_scores) / len(stat_true_scores)}")
                        print(f"stat false scores:   {sum(stat_false_scores) / len(stat_false_scores)}")
                        break
            neural_model.train()



if __name__ == "__main__":
    main()
