import math

import torch
from torch import nn, optim
from torch.functional import F
from torchtext import data

from transformers import BertConfig, BertModel

MINF = torch.log(torch.tensor(0.))


class EditDistBase(nn.Module):
    def __init__(self, ar_vocab, en_vocab, start_symbol, end_symbol):
        super(EditDistBase, self).__init__()

        self.ar_bos = ar_vocab[start_symbol]
        self.ar_eos = ar_vocab[end_symbol]
        self.en_bos = en_vocab[start_symbol]
        self.en_eos = en_vocab[end_symbol]

        self.ar_symbol_count = len(ar_vocab)
        self.en_symbol_count = len(en_vocab)

        self.n_target_classes = (
            1 +  # special termination symbol
            self.ar_symbol_count +  # delete arabic
            self.en_symbol_count +  # insert english
            self.ar_symbol_count * self.en_symbol_count)  # substitute

    @property
    def insertion_start(self):
        return 1 + self.ar_symbol_count

    @property
    def insertion_end(self):
        return 1 + self.ar_symbol_count + self.en_symbol_count

    def _deletion_id(self, ar_char):
        return 1 + ar_char

    def _insertion_id(self, en_char):
        return self.insertion_start + en_char

    def _substitute_id(self, ar_char, en_char):
        subs_id = (1 + self.ar_symbol_count + self.en_symbol_count +
                   self.en_symbol_count * ar_char + en_char)
        assert subs_id < self.n_target_classes
        return subs_id

    def _subsitute_start_and_end(self, ar_char):
        return (
            self.insertion_end + self.en_symbol_count * ar_char,
            self.insertion_end + self.en_symbol_count * (ar_char + 1))


class EditDistNeuralModel(EditDistBase):
    def __init__(self, ar_vocab, en_vocab, directed=False,
                 start_symbol="<s>", end_symbol="</s>"):
        super(EditDistNeuralModel, self).__init__(
            ar_vocab, en_vocab, start_symbol, end_symbol)

        self.directed = directed
        self.ar_encoder = self._encoder_for_vocab(ar_vocab)
        self.en_encoder = self._encoder_for_vocab(en_vocab, directed=directed)

        self.projection = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(2 * 64, 64),
            nn.Dropout(0.1),
            nn.ReLU())
        self.action_projection = nn.Linear(64, self.n_target_classes)

        if self.directed:
            self.output_symbol_projection = nn.Linear(64, len(en_vocab))

    def _encoder_for_vocab(self, vocab, directed=False):
        config = BertConfig(
            vocab_size=len(vocab),
            is_decoder=directed,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=128,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1)

        return BertModel(config)

    def _action_scores(self, ar_sent, en_sent, inference=False):
        # TODO masking when batched
        ar_len, en_len = ar_sent.size(1), en_sent.size(1)
        ar_vectors = self.ar_encoder(ar_sent)[0]

        if self.directed:
            en_vectors = self.en_encoder(
                en_sent, encoder_hidden_states=ar_vectors)[0]
        else:
            en_vectors = self.en_encoder(en_sent)[0]

        # TODO when batched, we will need an extra dimension for batch
        feature_table = self.projection(torch.cat((
            ar_vectors.transpose(0, 1).repeat(1, en_len, 1),
            en_vectors.repeat(ar_len, 1, 1)), dim=2))
        action_scores = F.log_softmax(
            self.action_projection(feature_table), dim=2)

        return ar_len, en_len, feature_table, action_scores

    def _symbol_prediction(self, feature_table, alpha):
        alpha_distributions = (alpha - alpha.logsumexp(1, keepdim=True)).exp()
        context_vector = (alpha_distributions.unsqueeze(2)
                          * feature_table).sum(1)

        return F.log_softmax(self.output_symbol_projection(feature_table.mean(0)), dim=1)

    def _forward_evaluation(self, ar_sent, en_sent, action_scores):
        """Differentiable forward pass through the model."""
        plausible_deletions = torch.zeros_like(action_scores) + MINF
        plausible_insertions = torch.zeros_like(action_scores) + MINF
        plausible_substitutions = torch.zeros_like(action_scores) + MINF

        alpha = []
        for t, ar_char in enumerate(ar_sent[0]):
            alpha.append([])
            for v, en_char in enumerate(en_sent[0]):
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
                if v >= 1:  # INSERTION
                    to_sum.append(
                        action_scores[t, v, insertion_id] + alpha[t][v - 1])
                if t >= 1:  # DELETION
                    to_sum.append(
                        action_scores[t, v, deletion_id] + alpha[t - 1][v])
                if v >= 1 and t >= 1:  # SUBSTITUTION
                    to_sum.append(
                        action_scores[t, v, subsitute_id] + alpha[t - 1][v - 1])
                    subs_from, subs_to = self._subsitute_start_and_end(ar_char)

                if not to_sum:
                    alpha[t].append(MINF)
                if len(to_sum) == 1:
                    alpha[t].append(to_sum[0])
                else:
                    alpha[t].append(torch.stack(to_sum).logsumexp(0))

        alpha_tensor = torch.stack([torch.stack(v) for v in alpha])
        return (
            alpha_tensor,
            plausible_deletions,
            plausible_insertions,
            plausible_substitutions)

    def forward(self, ar_sent, en_sent):
        ar_len, en_len, feature_table, action_scores = self._action_scores(
            ar_sent, en_sent)
        action_entropy = -(action_scores * action_scores.exp()).sum()

        (alpha, plausible_deletions, plausible_insertions,
         plausible_substitutions) = self._forward_evaluation(
            ar_sent, en_sent, action_scores)

        # This is the backward pass through the edit distance table.
        # Unlike, the forward pass it does not have to be differentiable.
        with torch.no_grad():
            beta = torch.zeros((ar_len, en_len)) + torch.log(torch.tensor(0.))
            beta[-1, -1] = 0.0

            for t, ar_char in reversed(list(enumerate(ar_sent[0]))):
                for v, en_char in reversed(list(enumerate(en_sent[0]))):
                    deletion_id = self._deletion_id(ar_char)
                    insertion_id = self._insertion_id(en_char)
                    subsitute_id = self._substitute_id(ar_char, en_char)

                    to_sum = [beta[t, v]]
                    if v < en_len - 1:
                        to_sum.append(
                            action_scores[t, v + 1, insertion_id] + beta[t, v + 1])
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

        next_symbol_scores = None
        if self.directed:
            next_symbol_scores = self._symbol_prediction(feature_table, alpha)

        return (action_scores, torch.exp(expected_counts), action_entropy,
                alpha[-1, -1], next_symbol_scores)

    def viterbi(self, ar_sent, en_sent):
        ar_len, en_len, _, action_scores = self._action_scores(
            ar_sent, en_sent)

        with torch.no_grad():
            action_count = torch.zeros((ar_len, en_len))
            alpha = torch.zeros((ar_len, en_len)) + MINF
            alpha[0, 0] = 0
            for t, ar_char in enumerate(ar_sent[0]):
                for v, en_char in enumerate(en_sent[0]):
                    if t == 0 and v == 0:
                        continue

                    deletion_id = self._deletion_id(ar_char)
                    insertion_id = self._insertion_id(en_char)
                    subsitute_id = self._substitute_id(ar_char, en_char)

                    possible_actions = []

                    if v >= 1:
                        possible_actions.append(
                            (action_scores[t, v, insertion_id] + alpha[t, v - 1],
                             action_count[t, v - 1] + 1))
                    if t >= 1:
                        possible_actions.append(
                            (action_scores[t, v, deletion_id] + alpha[t - 1, v],
                             action_count[t - 1, v] + 1))
                    if v >= 1 and t >= 1:
                        possible_actions.append(
                            (action_scores[t, v, subsitute_id] + alpha[t - 1, v - 1],
                             action_count[t - 1, v - 1] + 1))

                    best_action_cost, best_action_count = max(
                        possible_actions, key=lambda x: x[0] / x[1])

                    alpha[t, v] = best_action_cost
                    action_count[t, v] = best_action_count

        return torch.exp(alpha[-1, -1] / action_count[-1, -1])

    def decode(self, ar_sent):
        en_sent = torch.tensor([[self.en_bos]])
        ar_len, en_len, feature_table, action_scores = self._action_scores(
            ar_sent, en_sent, inference=False)

        alpha = None
        for v in range(2 * ar_sent.size(1)):
            if alpha is None:
                alpha = torch.zeros((ar_len, 1))
            else:
                alpha = torch.cat(
                    (alpha, torch.zeros(ar_len, 1) + MINF), dim=1)

            ar_len, en_len, feature_table, action_scores = self._action_scores(
                ar_sent, en_sent, inference=True)

            for t, ar_char in enumerate(ar_sent[0]):
                if t == 0 and v == 0:
                    alpha[0, 0] = 0.
                    continue

                deletion_id = self._deletion_id(ar_char)
                insertion_id = self._insertion_id(en_sent[0, v])
                subsitute_id = self._substitute_id(ar_char, en_sent[0, v])

                to_sum = []
                if v >= 1:
                    to_sum.append(
                        action_scores[t, v - 1, insertion_id] + alpha[t][v - 1])
                if t >= 1:
                    to_sum.append(
                        action_scores[t, v - 1, deletion_id] + alpha[t - 1][v])
                if t >= 1 and v >= 1:
                    to_sum.append(
                        action_scores[t, v - 1, subsitute_id] + alpha[t - 1][v - 1])

                if len(to_sum) == 1:
                    alpha[t, v] = to_sum[0]
                else:
                    alpha[t, v] = torch.stack(to_sum).logsumexp(0)

            # decide what the next symbol will be
            next_symbol_scores = self._symbol_prediction(
                feature_table[:, -1:], alpha[:, -1:])
            next_symbol = next_symbol_scores.argmax(1)

            # expand the target sequence
            en_sent = torch.cat(
                (en_sent, next_symbol.unsqueeze(0)), dim=1)
            if next_symbol == self.en_eos:
                break

        return en_sent


class EditDistStatModel(EditDistBase):
    def __init__(self, ar_vocab, en_vocab, start_symbol="<s>", end_symbol="</s>"):
        super(EditDistStatModel, self).__init__(
            ar_vocab, en_vocab, start_symbol, end_symbol)

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
                    to_sum.append(
                        self.weights[subsitute_id] + alpha[t - 1, v - 1])

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
        epsilon = torch.log(torch.tensor(1e-16)) + \
            torch.zeros_like(self.weights)
        expecation_sum = torch.stack([epsilon] + expectations).logsumexp(0)
        distribution = (
            expecation_sum - expecation_sum.logsumexp(0, keepdim=True))

        self.weights = torch.stack([
            torch.log(torch.tensor(0.9)) + self.weights,
            torch.log(torch.tensor(0.1)) + distribution]).logsumexp(0)
