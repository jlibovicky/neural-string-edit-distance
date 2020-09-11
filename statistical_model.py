
import torch

from models import EditDistBase, MINF


class EditDistStatModel(EditDistBase):
    """The original statistical algorithm by Ristad and Yanilos.

    This is a reimplemntation of the original learnable edit distance algorithm
    in PyTorch using the same interface as the neural models.
    """
    def __init__(self, ar_vocab, en_vocab, start_symbol="<s>",
                 end_symbol="</s>", pad_symbol="<pad>",
                 identitiy_initialize=True):
        super().__init__(
            ar_vocab, en_vocab, start_symbol, end_symbol, pad_symbol)

        weights = torch.tensor([
            1 / self.n_target_classes for _ in range(self.n_target_classes)])

        if identitiy_initialize:
            idenity_weight = [0. for _ in range(self.n_target_classes)]
            id_count = 0
            for idx, symbol in enumerate(self.ar_vocab.itos):
                if symbol in self.en_vocab.stoi:
                    idenity_weight[self._substitute_id(
                        idx, self.en_vocab[symbol])] = 1.
                    id_count += 1
            idenity_weight_tensor = torch.tensor(idenity_weight) / id_count
            weights = (weights + idenity_weight_tensor) / 2

        self.weights = torch.log(weights)


    def _forward_evaluation(self, ar_sent, en_sent):
        ar_len, en_len = ar_sent.size(1), en_sent.size(1)
        alpha = torch.zeros((ar_len, en_len)) + MINF
        alpha[0, 0] = 0
        for t, ar_char in enumerate(ar_sent[0]):
            for v, en_char in enumerate(en_sent[0]):
                if t == 0 and v == 0:
                    continue

                deletion_id = self._deletion_id(ar_char)
                insertion_id = self._insertion_id(en_char)
                subsitute_id = self._substitute_id(ar_char, en_char)

                to_sum = [alpha[t, v]]
                if v >= 1:
                    to_sum.append(self.weights[insertion_id] + alpha[t, v - 1])
                if t >= 1:
                    to_sum.append(self.weights[deletion_id] + alpha[t - 1, v])
                if v >= 1 and t >= 1:
                    to_sum.append(
                        self.weights[subsitute_id] + alpha[t - 1, v - 1])

                alpha[t, v] = torch.logsumexp(torch.tensor(to_sum), dim=0)
        return alpha

    def forward(self, ar_sent, en_sent):
        ar_len, en_len = ar_sent.size(1), en_sent.size(1)
        table_shape = ((ar_len, en_len, self.n_target_classes))

        alpha = self._forward_evaluation(ar_sent, en_sent)

        plausible_deletions = torch.zeros(table_shape) + MINF
        plausible_insertions = torch.zeros(table_shape) + MINF
        plausible_substitutions = torch.zeros(table_shape) + MINF

        beta = torch.zeros((ar_len, en_len)) + MINF
        beta[-1, -1] = 0.0

        for t in reversed(range(ar_len)):
            for v in reversed(range(en_len)):
                to_sum = [beta[t, v]]
                if v < en_len - 1:
                    insertion_id = self._insertion_id(en_sent[0, v + 1])
                    plausible_insertions[t, v, insertion_id] = 0
                    to_sum.append(
                        self.weights[insertion_id] + beta[t, v + 1])
                if t < ar_len - 1:
                    deletion_id = self._deletion_id(ar_sent[0, t + 1])
                    plausible_deletions[t, v, deletion_id] = 0
                    to_sum.append(
                        self.weights[deletion_id] + beta[t + 1, v])
                if v < en_len - 1 and t < ar_len - 1:
                    subsitute_id = self._substitute_id(
                        ar_sent[0, t + 1], en_sent[0, v + 1])
                    plausible_substitutions[t, v, subsitute_id] = 0
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

        action_count = torch.zeros((ar_len, en_len))
        alpha = torch.zeros((ar_len, en_len)) - MINF
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

    def maximize_expectation(self, expectations, learning_rate=0.1):
        assert 0 < learning_rate <= 1.0
        epsilon = torch.log(torch.tensor(1e-16)) + \
            torch.zeros_like(self.weights)
        expecation_sum = torch.stack([epsilon] + expectations).logsumexp(0)
        distribution = (
            expecation_sum - expecation_sum.logsumexp(0, keepdim=True))

        self.weights = torch.stack([
            torch.log(torch.tensor(1 - learning_rate)) + self.weights,
            torch.log(torch.tensor(learning_rate)) + distribution]).logsumexp(0)

    def decode(self, ar_sent, max_len=100, samples=10):
        assert samples > 0, "With zero samples nothing can be decoded."
        best_en_sent = []
        best_en_sent_score = MINF

        for _ in range(samples):
            en_sent = []
            ar_pos = 1
            ar_char = ar_sent[0, ar_pos]

            for _ in range(max_len):
                if ar_pos == ar_sent.size(1) - 1:
                    break

                ar_char = ar_sent[0, ar_pos]
                insert_weights = [
                    self.weights[self._insertion_id(i)]
                    for i in range(self.en_symbol_count)]
                delete_weight = [self.weights[self._deletion_id(ar_char)]]
                substitute_weight = [
                    self.weights[self._substitute_id(ar_char, i)]
                    for i in range(self.en_symbol_count)]

                distr = torch.tensor(insert_weights + substitute_weight + delete_weight).exp()
                distr /= distr.sum()
                next_op = torch.multinomial(distr, 1)[0]

                if next_op == 2 * self.en_symbol_count:
                    ar_pos += 1 # DELETE OP
                    continue

                en_sent.append(next_op % self.en_symbol_count)
                if next_op > self.en_symbol_count:
                    ar_pos += 1 # IT WAS A SUBSTITUTION

            if not en_sent:
                continue

            score = self._forward_evaluation(ar_sent, torch.tensor([en_sent]))[-1, -1]
            if score > best_en_sent_score:
                best_en_sent_score = score
                best_en_sent = en_sent

        return en_sent
