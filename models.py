import math

import torch
from torch import nn, optim
from torch.functional import F
from torchtext import data

from transformers import BertConfig, BertModel

MINF = torch.log(torch.tensor(0.))


class EditDistBase(nn.Module):
    def __init__(self, ar_vocab, en_vocab, start_symbol,
                 end_symbol, pad_symbol, full_table=True, tiny_table=False):
        super(EditDistBase, self).__init__()

        self.ar_bos = ar_vocab[start_symbol]
        self.ar_eos = ar_vocab[end_symbol]
        self.ar_pad = ar_vocab[pad_symbol]
        self.en_bos = en_vocab[start_symbol]
        self.en_eos = en_vocab[end_symbol]
        self.en_pad = en_vocab[pad_symbol]

        self.ar_symbol_count = len(ar_vocab)
        self.en_symbol_count = len(en_vocab)

        if full_table and tiny_table:
            raise ValueError(
                "Cannot use full table and tiny table at the same time.")

        self.full_table = full_table
        self.tiny_table = tiny_table

        if full_table:
            self.n_target_classes = (
                1 +  # special termination symbol
                self.ar_symbol_count +  # delete source
                self.en_symbol_count +  # insert target
                self.ar_symbol_count * self.en_symbol_count)  # substitute
        elif tiny_table:
            self.n_target_classes = 4
        else:
            self.n_target_classes = 1 + 2 * self.en_symbol_count

    def _deletion_id(self, ar_char):
        if self.full_table:
            return 1 + ar_char
        if self.tiny_table:
            return 0
        else:
            return 0

    def _insertion_id(self, en_char):
        if self.full_table:
            return 1 + self.ar_symbol_count + en_char
        if self.tiny_table:
            return 1
        else:
            return 1 + en_char

    def _substitute_id(self, ar_char, en_char):
        if self.full_table:
            subs_id = (1 + self.ar_symbol_count + self.en_symbol_count +
                       self.en_symbol_count * ar_char + en_char)
            assert subs_id < self.n_target_classes
            return subs_id
        if self.tiny_table:
            return 2
        else:
            return 1 + self.en_symbol_count + en_char


class EditDistNeuralModelConcurrent(EditDistBase):
    def __init__(self, ar_vocab, en_vocab, directed=False,
                 hidden_dim=32, hidden_layers=2, attention_heads=4,
                 start_symbol="<s>", end_symbol="</s>", pad_symbol="<pad>"):
        super(EditDistNeuralModelConcurrent, self).__init__(
            ar_vocab, en_vocab, start_symbol, end_symbol, pad_symbol,
            full_table=False, tiny_table=True)

        self.directed = directed
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.attention_heads = attention_heads
        self.ar_encoder = self._encoder_for_vocab(ar_vocab)
        self.en_encoder = self._encoder_for_vocab(en_vocab, directed=directed)

        self.projection = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(0.3),
            nn.ReLU())
        self.action_projection = nn.Linear(hidden_dim, self.n_target_classes)

    def _encoder_for_vocab(self, vocab, directed=False):
        config = BertConfig(
            vocab_size=len(vocab),
            is_decoder=directed,
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.hidden_layers,
            num_attention_heads=self.attention_heads,
            intermediate_size=2 * self.hidden_dim,
            hidden_act='gelu',
            hidden_dropout_prob=0.3,
            attention_probs_dropout_prob=0.3)

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

        feature_table = self.projection(torch.cat((
            ar_vectors.unsqueeze(2).repeat(1, 1, en_len, 1),
            en_vectors.unsqueeze(1).repeat(1, ar_len, 1, 1)), dim=3))
        action_scores = F.log_softmax(
            self.action_projection(feature_table), dim=3)

        return ar_len, en_len, feature_table, action_scores

    def _forward_evaluation(self, ar_sent, en_sent, action_scores):
        """Differentiable forward pass through the model."""
        alpha = []
        batch_size = ar_sent.size(0)
        b_range = torch.arange(batch_size)
        for t, ar_char in enumerate(ar_sent.transpose(0, 1)):
            alpha.append([])
            for v, en_char in enumerate(en_sent.transpose(0, 1)):
                if t == 0 and v == 0:
                    alpha[0].append(torch.zeros((batch_size,)))
                    continue

                deletion_id = self._deletion_id(ar_char)
                insertion_id = self._insertion_id(en_char)
                subsitute_id = self._substitute_id(ar_char, en_char)

                to_sum = []
                if v >= 1:  # INSERTION
                    to_sum.append(
                        action_scores[b_range, t, v, insertion_id] + alpha[t][v - 1])
                if t >= 1:  # DELETION
                    to_sum.append(
                        action_scores[b_range, t, v, deletion_id] + alpha[t - 1][v])
                if v >= 1 and t >= 1:  # SUBSTITUTION
                    to_sum.append(
                        action_scores[b_range, t, v, subsitute_id] + alpha[t - 1][v - 1])

                if not to_sum:
                    alpha[t].append(torch.zeros((batch_size,)) + MINF)
                if len(to_sum) == 1:
                    alpha[t].append(to_sum[0])
                else:
                    alpha[t].append(
                        torch.stack(to_sum).logsumexp(0))

        alpha_tensor = torch.stack(
            [torch.stack(v) for v in alpha]).permute(2, 0, 1)
        return alpha_tensor

    # TODO add @torch.no_grad()
    def _backward_evalatuion_and_expectation(
            self, ar_len, en_len, ar_sent, en_sent, alpha, action_scores):
        # This is the backward pass through the edit distance table.
        # Unlike, the forward pass it does not have to be differentiable.
        plausible_deletions = torch.zeros_like(action_scores) + MINF
        plausible_insertions = torch.zeros_like(action_scores) + MINF
        plausible_substitutions = torch.zeros_like(action_scores) + MINF

        batch_size = ar_sent.size(0)
        b_range = torch.arange(batch_size)

        with torch.no_grad():
            beta = torch.zeros_like(alpha) + MINF
            beta[:, -1, -1] = 0.0

            for t, ar_char in reversed(list(enumerate(ar_sent.transpose(0, 1)))):
                for v, en_char in reversed(list(enumerate(en_sent.transpose(0, 1)))):
                    deletion_id = self._deletion_id(ar_char)
                    insertion_id = self._insertion_id(en_char)
                    subsitute_id = self._substitute_id(ar_char, en_char)

                    to_sum = [beta[:, t, v]]
                    if v < en_len - 1:
                        plausible_insertions[b_range, t, v, insertion_id] = 0
                        to_sum.append(
                            action_scores[b_range, t, v + 1, insertion_id] + beta[:, t, v + 1])
                    if t < ar_len - 1:
                        plausible_deletions[b_range, t, v, deletion_id] = 0
                        to_sum.append(
                            action_scores[b_range, t + 1, v, deletion_id] + beta[:, t + 1, v])
                    if v < en_len - 1 and t < ar_len - 1:
                        plausible_substitutions[b_range, t, v, subsitute_id] = 0
                        to_sum.append(
                            action_scores[b_range, t + 1, v + 1, subsitute_id] + beta[:, t + 1, v + 1])

                    beta[:, t, v] = torch.stack(to_sum).logsumexp(0)

            # deletion expectation
            expected_deletions = torch.zeros_like(action_scores) + MINF
            expected_deletions[: ,1:, :] = (
                alpha[:, :-1, :].unsqueeze(3) +
                action_scores[:, 1:, :] + plausible_deletions[:, 1:, :] +
                beta[:, 1:, :].unsqueeze(3))
            # insertions expectation
            expected_insertions = torch.zeros_like(action_scores) + MINF
            expected_insertions[:, :, 1:] = (
                alpha[:, :, :-1].unsqueeze(3) +
                action_scores[:, :, 1:] + plausible_insertions[:, :, 1:] +
                beta[:, :, 1:].unsqueeze(3))
            # substitution expectation
            expected_substitutions = torch.zeros_like(action_scores) + MINF
            expected_substitutions[:, 1:, 1:] = (
                alpha[:, :-1, :-1].unsqueeze(3) +
                action_scores[:, 1:, 1:] + plausible_substitutions[:, 1:, 1:] +
                beta[:, 1:, 1:].unsqueeze(3))

            expected_counts = torch.stack([
                expected_deletions, expected_insertions, expected_substitutions]).logsumexp(0)
            expected_counts -= expected_counts.logsumexp(3, keepdim=True)
        return expected_counts

    def forward(self, ar_sent, en_sent):
        ar_len, en_len, feature_table, action_scores = self._action_scores(
            ar_sent, en_sent)

        alpha = self._forward_evaluation(ar_sent, en_sent, action_scores)
        expected_counts = self._backward_evalatuion_and_expectation(
            ar_len, en_len, ar_sent, en_sent, alpha, action_scores)

        return (action_scores, torch.exp(expected_counts), alpha[0, -1, -1])

    @torch.no_grad()
    def viterbi(self, ar_sent, en_sent):
        ar_len, en_len, _, action_scores = self._action_scores(
            ar_sent, en_sent)
        action_scores = action_scores.squeeze(0)

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


class EditDistNeuralModelProgressive(EditDistNeuralModelConcurrent):
    def __init__(self, ar_vocab, en_vocab, directed=False,
                 hidden_dim=32, hidden_layers=2, attention_heads=4,
                 start_symbol="<s>", end_symbol="</s>", pad_symbol="<pad>"):
        super(EditDistNeuralModelProgressive, self).__init__(
            ar_vocab, en_vocab, directed, full_table=False,
            hidden_dim=hidden_dim, hidden_layers=hidden_layers,
            attention_heads=attention_heads,
            start_symbol=start_symbol, end_symbol=end_symbol, pad_symbol=pad_symbol)

        self.deletion_logit_proj = nn.Linear(self.hidden_dim, 1)
        self.insertion_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.substitution_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.tgt_embeddings = \
            self.en_encoder.embeddings.word_embeddings.weight.t()

    def _action_scores(self, ar_sent, en_sent, inference=False):
        # TODO masking when batched
        ar_len, en_len = ar_sent.size(1), en_sent.size(1)
        ar_vectors = self.ar_encoder(ar_sent)[0]

        if self.directed:
            # shift the en_sent even one more
            en_vectors = self.en_encoder(
                en_sent, encoder_hidden_states=ar_vectors)[0]
        else:
            en_vectors = self.en_encoder(en_sent)[0]

        feature_table = self.projection(torch.cat((
            ar_vectors.unsqueeze(2).repeat(1, 1, en_len, 1),
            en_vectors.unsqueeze(1).repeat(1, ar_len, 1, 1)), dim=3))

        # DELETION <<<
        valid_deletion_logits = self.deletion_logit_proj(feature_table[:, :-1])
        deletion_padding = torch.zeros_like(valid_deletion_logits[:, :1]) + MINF
        padded_deletion_logits = torch.cat(
            (deletion_padding, valid_deletion_logits), dim=1)

        # INSERTIONS <<<
        valid_insertion_logits = torch.matmul(
            self.insertion_proj(feature_table[:, :, :-1]),
            self.tgt_embeddings)
        insertion_padding = torch.zeros((
            valid_insertion_logits.size(0),
            valid_insertion_logits.size(1), 1,
            valid_insertion_logits.size(3))) + MINF
        padded_insertion_logits = torch.cat(
            (insertion_padding, valid_insertion_logits), dim=2)

        # SUBSITUTION <<<
        valid_subs_logits = torch.matmul(
            self.substitution_proj(feature_table[:, :-1, :-1]),
            self.tgt_embeddings)
        src_subs_padding = torch.zeros_like(valid_subs_logits[:, :1]) + MINF
        src_padded_subs_logits = torch.cat(
            (src_subs_padding, valid_subs_logits), dim=1)
        tgt_subs_padding = torch.zeros((
            src_padded_subs_logits.size(0),
            src_padded_subs_logits.size(1), 1,
            src_padded_subs_logits.size(3))) + MINF
        padded_subs_logits = torch.cat(
            (tgt_subs_padding, src_padded_subs_logits), dim=2)

        action_scores = F.log_softmax(torch.cat(
            (padded_deletion_logits, padded_insertion_logits, padded_subs_logits),
            dim=3), dim=3)

        assert action_scores.size(1) == ar_len
        assert action_scores.size(2) == en_len
        assert action_scores.size(3) == self.n_target_classes

        return (ar_len, en_len, en_vectors, feature_table.squeeze(0), action_scores,
                valid_insertion_logits,
                valid_subs_logits)

    def forward(self, ar_sent, en_sent):
        (ar_len, en_len, en_states, feature_table, action_scores,
            insertion_logits, subs_logits) = self._action_scores(
                ar_sent, en_sent)

        alpha = self._forward_evaluation(ar_sent, en_sent, action_scores)
        expected_counts = self._backward_evalatuion_and_expectation(
            ar_len, en_len, ar_sent, en_sent, alpha, action_scores)

        insertion_log_dist = (F.log_softmax(insertion_logits, dim=3) +
                              alpha[:, :, :-1].unsqueeze(3))
        subs_log_dist = (F.log_softmax(subs_logits, dim=3) +
                         alpha[:, :-1, :-1].unsqueeze(3))

        next_symbol_logprobs_sum = torch.cat(
            (insertion_log_dist, subs_log_dist), dim=1).logsumexp(1)
        next_symbol_logprobs = (
            next_symbol_logprobs_sum -
            next_symbol_logprobs_sum.logsumexp(2, keepdims=True))

        seq2seq_logits = torch.matmul(en_states, self.tgt_embeddings)

        return (action_scores, torch.exp(expected_counts),
                alpha[-1, -1], next_symbol_logprobs, seq2seq_logits)

    def decode(self, ar_sent):
        en_sent = torch.tensor([[self.en_bos]])
        (ar_len, en_len, _, feature_table,
         action_scores, _, _) = self._action_scores(
            ar_sent, en_sent, inference=False)
        action_scores = action_scores.squeeze(0)

        # special case, v = 0
        alpha = torch.zeros((ar_len, 1)) + MINF
        for t, ar_char in enumerate(ar_sent[0]):
            if t == 0:
                alpha[0, 0] = 0.
                continue
            deletion_id = self._deletion_id(ar_char)
            alpha[t, 0] = action_scores[t, 0, deletion_id] + alpha[t - 1][0]

        for v in range(1, 2 * ar_sent.size(1)):
            # From what we have, do a prediction what is the next symbol
            insertion_logits = torch.matmul(
                self.insertion_proj(feature_table[:, v - 1:v]),
                self.tgt_embeddings)# + alpha[:, v - 1:v].unsqueeze(2)
            # TODO maybe weight by alpha?

            subs_logits = torch.matmul(
                self.substitution_proj(feature_table[1:, v - 1:v]),
                self.tgt_embeddings)# + alpha[1:, v - 1:v].unsqueeze(2)
            # TODO maybe weight by alpha?

            # TODO and what about summing them instead of logsumexping?
            next_symb_logits = torch.cat(
                (insertion_logits, subs_logits), dim=0).logsumexp(0)
            next_symbol = next_symb_logits.argmax(1)

            en_sent = torch.cat(
                (en_sent, next_symbol.unsqueeze(0)), dim=1)

            ar_len, en_len, _, feature_table, action_scores, _, _ = self._action_scores(
                ar_sent, en_sent, inference=True)
            action_scores = action_scores.squeeze(0)
            alpha = torch.cat(
                (alpha, torch.zeros(ar_len, 1) + MINF), dim=1)

            for t, ar_char in enumerate(ar_sent[0]):
                deletion_id = self._deletion_id(ar_char)
                insertion_id = self._insertion_id(en_sent[0, v])
                subsitute_id = self._substitute_id(ar_char, en_sent[0, v])

                to_sum = [
                    action_scores[t, v - 1, insertion_id] + alpha[t][v - 1]]
                if t >= 1:
                    to_sum.append(
                        action_scores[t, v - 1, deletion_id] + alpha[t - 1][v])
                    to_sum.append(
                        action_scores[t, v - 1, subsitute_id] + alpha[t - 1][v - 1])

                if len(to_sum) == 1:
                    alpha[t, v] = to_sum[0]
                else:
                    alpha[t, v] = torch.stack(to_sum).logsumexp(0)

            # expand the target sequence
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

        return (action_scores, torch.exp(expected_counts),
                alpha[-1, -1], next_symbol_logprobs)

    def decode(self, ar_sent):
        en_sent = torch.tensor([[self.en_bos]])
        (ar_len, en_len, feature_table,
         action_scores, _, _) = self._action_scores(
            ar_sent, en_sent, inference=False)

        # special case, v = 0
        alpha = torch.zeros((ar_len, 1)) + MINF
        for t, ar_char in enumerate(ar_sent[0]):
            if t == 0:
                alpha[0, 0] = 0.
                continue
            deletion_id = self._deletion_id(ar_char)
            alpha[t, 0] = action_scores[t, 0, deletion_id] + alpha[t - 1][0]

        for v in range(1, 2 * ar_sent.size(1)):
            # From what we have, do a prediction what is the next symbol
            insertion_logits = torch.matmul(
                self.insertion_proj(feature_table[:, v - 1:v]),
                self.tgt_embeddings)  # + alpha[:, v - 1:v].unsqueeze(2)
            # TODO maybe weight by alpha?

            subs_logits = torch.matmul(
                self.substitution_proj(feature_table[1:, v - 1:v]),
                self.tgt_embeddings)  # + alpha[1:, v - 1:v].unsqueeze(2)
            # TODO maybe weight by alpha?

            # TODO and what about summing them instead of logsumexping?
            next_symb_logits = torch.cat(
                (insertion_logits, subs_logits), dim=0).logsumexp(0)
            next_symbol = next_symb_logits.argmax(1)

            en_sent = torch.cat(
                (en_sent, next_symbol.unsqueeze(0)), dim=1)

            ar_len, en_len, feature_table, action_scores, _, _ = self._action_scores(
                ar_sent, en_sent, inference=True)
            alpha = torch.cat(
                (alpha, torch.zeros(ar_len, 1) + MINF), dim=1)

            for t, ar_char in enumerate(ar_sent[0]):
                deletion_id = self._deletion_id(ar_char)
                insertion_id = self._insertion_id(en_sent[0, v])
                subsitute_id = self._substitute_id(ar_char, en_sent[0, v])

                to_sum = [
                    action_scores[t, v - 1, insertion_id] + alpha[t][v - 1]]
                if t >= 1:
                    to_sum.append(
                        action_scores[t, v - 1, deletion_id] + alpha[t - 1][v])
                    to_sum.append(
                        action_scores[t, v - 1, subsitute_id] + alpha[t - 1][v - 1])

                if len(to_sum) == 1:
                    alpha[t, v] = to_sum[0]
                else:
                    alpha[t, v] = torch.stack(to_sum).logsumexp(0)

            # expand the target sequence
            if next_symbol == self.en_eos:
                break

        return en_sent


class EditDistStatModel(EditDistBase):
    def __init__(self, ar_vocab, en_vocab, start_symbol="<s>",
                 end_symbol="</s>", pad_symbol="<pad>"):
        super(EditDistStatModel, self).__init__(
            ar_vocab, en_vocab, start_symbol, end_symbol, pad_symbol)

        self.weights = torch.log(torch.tensor(
            [1 / self.n_target_classes for _ in range(self.n_target_classes)]))

    def forward(self, ar_sent, en_sent):
        ar_len, en_len = ar_sent.size(1), en_sent.size(1)
        table_shape = ((ar_len, en_len, self.n_target_classes))

        plausible_deletions = torch.zeros(table_shape) + MINF
        plausible_insertions = torch.zeros(table_shape) + MINF
        plausible_substitutions = torch.zeros(table_shape) + MINF

        ar_sent = ar_sent.transpose(0, 1)
        en_sent = en_sent.transpose(0, 1)

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
        ar_len, en_len = ar_sent.size(1), en_sent.size(1)

        ar_sent = ar_sent.transpose(0, 1)
        en_sent = en_sent.transpose(0, 1)

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
