"""Neural string edit distance."""

import math

import torch
from torch import nn, optim
from torch.functional import F
from torchtext import data

from transformers import BertConfig, BertModel

from rnn import RNNEncoder, RNNDecoder
from cnn import CNNEncoder, CNNDecoder

MINF = torch.log(torch.tensor(0.))


class EditDistBase(nn.Module):
    """Base class used both for statistical and neural model."""
    def __init__(self, ar_vocab, en_vocab, start_symbol,
                 end_symbol, pad_symbol, table_type="full", extra_classes=0):
        super(EditDistBase, self).__init__()

        self.ar_vocab = ar_vocab
        self.en_vocab = en_vocab

        self.ar_bos = ar_vocab[start_symbol]
        self.ar_eos = ar_vocab[end_symbol]
        self.ar_pad = ar_vocab[pad_symbol]
        self.en_bos = en_vocab[start_symbol]
        self.en_eos = en_vocab[end_symbol]
        self.en_pad = en_vocab[pad_symbol]

        self.extra_classes = extra_classes
        self.ar_symbol_count = len(ar_vocab)
        self.en_symbol_count = len(en_vocab)

        self.table_type = table_type

        if table_type == "full":
            self.deletion_classes = self.ar_symbol_count
            self.insertion_classes = self.en_symbol_count
            self.subs_classes = self.ar_symbol_count * self.en_symbol_count
            self.n_target_classes = (
                self.extra_classes +
                self.ar_symbol_count +  # delete source
                self.en_symbol_count +  # insert target
                self.ar_symbol_count * self.en_symbol_count)  # substitute
        elif table_type == "tiny":
            self.deletion_classes = 1
            self.insertion_classes = 1
            self.subs_classes = 1
            self.n_target_classes = self.extra_classes + 3
        elif table_type == "vocab":
            self.deletion_classes = 1
            self.insertion_classes = self.en_symbol_count
            self.subs_classes = self.en_symbol_count
            self.n_target_classes = (
                self.extra_classes + 1 + 2 * self.en_symbol_count)
        else:
            raise ValueError("Unknown table type.")

    def _deletion_id(self, ar_char):
        if self.table_type == "full":
            return ar_char
        return torch.zeros_like(ar_char)

    def _insertion_id(self, en_char):
        if self.table_type == "full":
            return self.ar_symbol_count + en_char
        elif self.table_type == "tiny":
            return 1
        elif self.table_type == "vocab":
            return 1 + en_char
        raise RuntimeError("Unknown table type.")

    def _substitute_id(self, ar_char, en_char):
        if self.table_type == "full":
            subs_id = (self.ar_symbol_count + self.en_symbol_count +
                       self.en_symbol_count * ar_char + en_char)
            assert subs_id < self.n_target_classes
            return subs_id
        elif self.table_type == "tiny":
            return 2
        elif self.table_type == "vocab":
            return 1 + self.en_symbol_count + en_char
        raise RuntimeError("Unknown table type.")


def get_distortion_mask(max_size=512):
    """Mask to be applied on alpha during training.

    The purpose of the distrotion maks is to dicourage the model from making
    states far from diagonal too probable. In other words, discourage the model
    from considering: delete everything and then insert everything to be a good
    output.
    """

    mask = []
    for i in range(max_size):
        row = []
        for j in range(max_size):
            row.append(max(0, abs(i - j) - 1))
        mask.append(row)
    return torch.tensor(mask).float().unsqueeze(0)


class NeuralEditDistBase(EditDistBase):
    """Base class for neural models.

    Implements the forward and bakward probability computation, the EM loss
    function and Viterbi decoding.
    """
    def __init__(self, ar_vocab, en_vocab, device, directed=False,
                 encoder_decoder_attention=True,
                 share_encoders=False,
                 table_type="vocab", extra_classes=0,
                 window=3,
                 hidden_dim=32, hidden_layers=2, attention_heads=4,
                 model_type="transformer",
                 start_symbol="<s>", end_symbol="</s>", pad_symbol="<pad>"):
        super(NeuralEditDistBase, self).__init__(
            ar_vocab, en_vocab, start_symbol, end_symbol, pad_symbol,
            table_type=table_type, extra_classes=extra_classes)

        if directed and share_encoders:
            raise ValueError(
                "You cannot share encoder if one of them is decoder.")
        if share_encoders:
            if ar_vocab != en_vocab:
                raise ValueError(
                    "When sharing encoders, vocabularies must be the same.")

        self.model_type = model_type
        self.device = device
        self.directed = directed
        self.hidden_dim = hidden_dim
        self.window = window
        if self.model_type == "bert":
            self.hidden_dim = 768
        self.hidden_layers = hidden_layers
        self.attention_heads = attention_heads
        self.ar_encoder = self._encoder_for_vocab(ar_vocab)
        if model_type == "bert" or share_encoders:
            self.en_encoder = self.ar_encoder
        else:
            self.en_encoder = self._encoder_for_vocab(en_vocab, directed=directed)

        self.projection = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(self.hidden_dim))

        self.encoder_decoder_attention = encoder_decoder_attention

        proj_source = 2 * self.hidden_dim if self.directed else self.hidden_dim

        self.deletion_logit_proj = nn.Linear(
            proj_source, self.deletion_classes)
        self.insertion_proj = nn.Linear(
            proj_source, self.insertion_classes)
        self.substitution_proj = nn.Linear(
            proj_source, self.subs_classes)
        self.extra_proj = nn.Linear(proj_source, self.extra_classes)

        self.distortion_mask = get_distortion_mask().to(device)


    def _encoder_for_vocab(self, vocab, directed=False):
        if self.model_type == "transformer":
            return self._transformer_for_vocab(vocab, directed)
        elif self.model_type == "rnn":
            return self._rnn_for_vocab(vocab, directed)
        elif self.model_type == "bert":
            return BertModel.from_pretrained("bert-base-cased")
        elif self.model_type == "embeddings":
            return self._cnn_for_vocab(vocab, directed, hidden=False)
        elif self.model_type == "cnn":
            return self._cnn_for_vocab(vocab, directed, hidden=True)
        raise ValueError(f"Uknown model type {self.model_type}.")

    def _transformer_for_vocab(self, vocab, directed=False):
        config = BertConfig(
            vocab_size=len(vocab),
            is_decoder=directed,
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.hidden_layers,
            num_attention_heads=self.attention_heads,
            intermediate_size=2 * self.hidden_dim,
            hidden_act='relu',
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1)

        return BertModel(config)

    def _rnn_for_vocab(self, vocab, directed=False):
        if not directed:
            return RNNEncoder(
                vocab, self.hidden_dim,
                self.hidden_dim, self.hidden_layers, dropout=0.1)
        else:
            return RNNDecoder(
                vocab, self.hidden_dim,
                self.hidden_dim, self.hidden_layers, self.attention_heads,
                output_proj=False, dropout=0.1)

    def _cnn_for_vocab(self, vocab, directed=False, hidden=True, dropout=0.1):
        layers = self.hidden_layers if hidden else 0

        if not directed:
            return CNNEncoder(
                vocab, self.hidden_dim,
                self.hidden_dim,
                window=self.window,
                layers=layers, dropout=dropout)
        else:
            return CNNDecoder(
                vocab, self.hidden_dim,
                self.hidden_dim, layers=layers,
                window=self.window,
                attention_heads=self.attention_heads,
                output_proj=False, dropout=dropout)

    def _encode_ar(self, inputs, mask):
        return self.ar_encoder(inputs, attention_mask=mask)[0]

    def _encode_en(self, inputs, mask, ar_vectors, ar_mask):
        if self.encoder_decoder_attention:
            return self.en_encoder(
                inputs, attention_mask=mask,
                encoder_hidden_states=ar_vectors,
                encoder_attention_mask=ar_mask)[0]
        return self.en_encoder(
            inputs, attention_mask=mask)[0]

    def _action_scores(self, ar_sent, en_sent, inference=False):
        """Compute possible action probabilities (Eq. 3 and 4)."""
        ar_len, en_len = ar_sent.size(1), en_sent.size(1)
        ar_mask = ar_sent != self.ar_pad
        en_mask = en_sent != self.en_pad
        ar_vectors = self._encode_ar(ar_sent, ar_mask)
        en_vectors = self._encode_en(en_sent, en_mask, ar_vectors, ar_mask)

        feature_table = self.projection(torch.cat((
            ar_vectors.unsqueeze(2).repeat(1, 1, en_len, 1),
            en_vectors.unsqueeze(1).repeat(1, ar_len, 1, 1)), dim=3))

        if self.directed:
            feature_table = torch.cat(
                (feature_table, en_vectors.unsqueeze(1).repeat(
                    1, ar_len, 1, 1)),
                dim=3)

        # DELETION <<<
        valid_deletion_logits = self.deletion_logit_proj(feature_table[:, :-1])
        deletion_padding = torch.full_like(valid_deletion_logits[:, :1], MINF)
        padded_deletion_logits = torch.cat(
            (deletion_padding, valid_deletion_logits), dim=1)

        # INSERTIONS <<<
        valid_insertion_logits = self.insertion_proj(feature_table[:, :, :-1])
        insertion_padding = torch.full((
            valid_insertion_logits.size(0),
            valid_insertion_logits.size(1), 1,
            valid_insertion_logits.size(3)), MINF).to(self.device)
        padded_insertion_logits = torch.cat(
            (insertion_padding, valid_insertion_logits), dim=2)

        # SUBSITUTION <<<
        valid_subs_logits = self.substitution_proj(feature_table[:, :-1, :-1])
        src_subs_padding = torch.full_like(valid_subs_logits[:, :1], MINF)
        src_padded_subs_logits = torch.cat(
            (src_subs_padding, valid_subs_logits), dim=1)
        tgt_subs_padding = torch.full((
            src_padded_subs_logits.size(0),
            src_padded_subs_logits.size(1), 1,
            src_padded_subs_logits.size(3)), MINF).to(self.device)
        padded_subs_logits = torch.cat(
            (tgt_subs_padding, src_padded_subs_logits), dim=2)

        actions_to_concat = [
            padded_deletion_logits, padded_insertion_logits,
            padded_subs_logits]

        if self.extra_classes > 0:
            extra_logits = self.extra_proj(feature_table)
            actions_to_concat.append(extra_logits)

        action_scores = F.log_softmax(torch.cat(
            actions_to_concat, dim=3), dim=3)

        assert action_scores.size(1) == ar_len
        assert action_scores.size(2) == en_len
        assert action_scores.size(3) == self.n_target_classes

        return (ar_len, en_len, feature_table, action_scores,
                valid_insertion_logits,
                valid_subs_logits)

    def _forward_evaluation(self, ar_sent, en_sent, action_scores):
        """Differentiable forward pass through the model. Algorithm 1."""
        alpha = []
        batch_size = ar_sent.size(0)
        b_range = torch.arange(batch_size).to(self.device)
        for t, ar_char in enumerate(ar_sent.transpose(0, 1)):
            alpha.append([])
            for v, en_char in enumerate(en_sent.transpose(0, 1)):
                if t == 0 and v == 0:
                    alpha[0].append(torch.zeros((batch_size,)).to(self.device))
                    continue

                deletion_id = self._deletion_id(ar_char)
                insertion_id = self._insertion_id(en_char)
                subsitute_id = self._substitute_id(ar_char, en_char)

                to_sum = []
                if v >= 1:  # INSERTION
                    to_sum.append(
                        action_scores[b_range, t, v, insertion_id] +
                        alpha[t][v - 1])
                if t >= 1:  # DELETION
                    to_sum.append(
                        action_scores[b_range, t, v, deletion_id] +
                        alpha[t - 1][v])
                if v >= 1 and t >= 1:  # SUBSTITUTION
                    to_sum.append(
                        action_scores[b_range, t, v, subsitute_id] +
                        alpha[t - 1][v - 1])

                if not to_sum:
                    alpha[t].append(
                        torch.full(batch_size, MINF).to(self.device))
                if len(to_sum) == 1:
                    alpha[t].append(to_sum[0])
                else:
                    alpha[t].append(
                        torch.stack(to_sum).logsumexp(0))

        alpha_tensor = torch.stack(
            [torch.stack(v) for v in alpha]).permute(2, 0, 1)
        return alpha_tensor

    @torch.no_grad()
    def _backward_evalatuion_and_expectation(
            self, ar_len, en_len, ar_sent, en_sent, alpha, action_scores):
        """The backward pass through the edit distance table. Algorithm 2.

        Unlike, the forward pass it does not have to be differentiable, because
        it is only used to compute the expected distribution that is as a
        "target" in the EM loss.
        """
        plausible_deletions = torch.full_like(action_scores, MINF)
        plausible_insertions = torch.full_like(action_scores, MINF)
        plausible_substitutions = torch.full_like(action_scores, MINF)

        ar_lengths = (ar_sent != self.ar_pad).sum(1)
        en_lengths = (en_sent != self.en_pad).sum(1)

        batch_size = ar_sent.size(0)
        b_range = torch.arange(batch_size)

        beta = torch.full_like(alpha, MINF)
        beta[:, -1, -1] = 0.0

        for t in reversed(range(ar_sent.size(1))):
            for v in reversed(range(en_sent.size(1))):
                # Bool mask: when we are in the table inside both words
                is_valid = (v <= (en_lengths - 1)) * (t <= (ar_lengths - 1))
                # Bool mask: true for end state of word pairs
                is_corner = (v == (en_lengths - 1)) * (t == (ar_lengths - 1))

                to_sum = [beta[:, t, v]]
                if v < en_len - 1:
                    insertion_id = self._insertion_id(en_sent[:, v])
                    plausible_insertions[b_range, t, v, insertion_id] = 0

                    # What would be the insertion score look like if there
                    # were anything to insert
                    insertion_score_candidate = (
                        action_scores[b_range, t, v, insertion_id] +
                        beta[:, t, v + 1])

                    # This keeps MINF before we get inside the words
                    to_sum.append(torch.where(
                        is_valid,
                        insertion_score_candidate,
                        torch.full_like(insertion_score_candidate, MINF)))
                if t < ar_len - 1:
                    deletion_id = self._deletion_id(ar_sent[:, t])
                    plausible_deletions[b_range, t, v, deletion_id] = 0

                    deletion_score_candidate = (
                        action_scores[b_range, t, v, deletion_id] +
                        beta[:, t + 1, v])

                    to_sum.append(torch.where(
                        is_valid,
                        deletion_score_candidate,
                        torch.full_like(deletion_score_candidate, MINF)))
                if v < en_len - 1 and t < ar_len - 1:
                    subsitute_id = self._substitute_id(
                        ar_sent[:, t], en_sent[:, v])
                    plausible_substitutions[
                        b_range, t, v, subsitute_id] = 0

                    substitution_score_candidate = (
                        action_scores[b_range, t, v, subsitute_id] +
                        beta[:, t + 1, v + 1])

                    to_sum.append(torch.where(
                        is_valid,
                        substitution_score_candidate,
                        torch.full_like(insertion_score_candidate, MINF)))

                beta_candidate = torch.stack(to_sum).logsumexp(0)

                beta[:, t, v] = torch.where(
                    is_corner, torch.zeros_like(beta_candidate),
                    torch.where(is_valid, beta_candidate,
                                torch.full_like(beta_candidate, MINF)))

        # deletion expectation
        expected_deletions = torch.zeros_like(action_scores) + MINF
        expected_deletions[:, 1:, :] = (
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
            expected_deletions, expected_insertions,
            expected_substitutions], dim=4).logsumexp(4)
        expected_counts -= expected_counts.logsumexp(3, keepdim=True)
        return beta, expected_counts

    def _alpha_distortion_penalty(self, src_len, tgt_len, alpha_table):
        """Penalty for the alphas being too high outside from the diagonal."""
        penalties = self.distortion_mask[:, :src_len, :tgt_len]
        return alpha_table.exp() * penalties

    @torch.no_grad()
    def viterbi(self, ar_sent, en_sent):
        """Get a single best sequence of edit ops for a string pair."""
        assert ar_sent.size(0) == 1
        ar_len, en_len, _, action_scores, _, _ = self._action_scores(
            ar_sent, en_sent)
        action_scores = action_scores.squeeze(0)

        action_count = torch.zeros((ar_len, en_len))
        actions = torch.zeros((ar_len, en_len))
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
                         action_count[t, v - 1] + 1, 0))
                else:
                    possible_actions.append((-1e12, 1.0, 0))
                if t >= 1:
                    possible_actions.append(
                        (action_scores[t, v, deletion_id] + alpha[t - 1, v],
                         action_count[t - 1, v] + 1, 1))
                else:
                    possible_actions.append((-1e12, 1.0, 1))
                if v >= 1 and t >= 1:
                    possible_actions.append(
                        (action_scores[t, v, subsitute_id] + alpha[t - 1, v - 1],
                         action_count[t - 1, v - 1] + 1, 2))
                else:
                    possible_actions.append((-1e12, 1.0, 2))

                best_action_cost, best_action_count, best_action_id = max(
                    possible_actions, key=lambda x: x[0])

                alpha[t, v] = best_action_cost
                action_count[t, v] = best_action_count
                actions[t, v] = best_action_id

        operations = []
        t = ar_len - 1
        v = en_len - 1
        while t > 0 or v > 0:
            if actions[t, v] == 1:
                operations.append(("delete", ar_sent[0, t - 1].cpu().numpy()))
                t -= 1
            elif actions[t, v] == 0:
                operations.append(("insert", en_sent[0, v].cpu().numpy()))
                v -= 1
            elif actions[t, v] == 2:
                operations.append(
                    ("subs",
                     (ar_sent[0, t - 1].cpu().numpy(),
                      en_sent[0, v].cpu().numpy())))
                v -= 1
                t -= 1
        operations.reverse()

        return (torch.exp(alpha[-1, -1] / action_count[-1, -1]),
                operations)


class EditDistNeuralModelConcurrent(NeuralEditDistBase):
    """Model for binary sequence-pari classification."""
    def __init__(self, ar_vocab, en_vocab, device, directed=False,
                 hidden_dim=32, hidden_layers=2, attention_heads=4,
                 share_encoders=False,
                 model_type="transformer",
                 start_symbol="<s>", end_symbol="</s>", pad_symbol="<pad>"):
        super(EditDistNeuralModelConcurrent, self).__init__(
            ar_vocab, en_vocab, device, directed,
            encoder_decoder_attention=False,
            share_encoders=share_encoders,
            hidden_dim=hidden_dim, hidden_layers=hidden_layers,
            attention_heads=attention_heads,
            table_type="tiny", extra_classes=1,
            start_symbol=start_symbol, end_symbol=end_symbol,
            pad_symbol=pad_symbol, model_type=model_type)

    def forward(self, ar_sent, en_sent):
        batch_size = ar_sent.size(0)
        b_range = torch.arange(batch_size)
        ar_lengths = (ar_sent != self.ar_pad).int().sum(1) - 1
        en_lengths = (en_sent != self.en_pad).int().sum(1) - 1
        ar_len, en_len, _, action_scores, _, _ = self._action_scores(
            ar_sent, en_sent)

        alpha = self._forward_evaluation(ar_sent, en_sent, action_scores)
        _, expected_counts = self._backward_evalatuion_and_expectation(
            ar_len, en_len, ar_sent, en_sent, alpha, action_scores)

        distorted_probs = self._alpha_distortion_penalty(
            ar_len, en_len, alpha)

        return (action_scores, torch.exp(expected_counts),
                alpha[b_range, ar_lengths, en_lengths], distorted_probs)

    @torch.no_grad()
    def probabilities(self, ar_sent, en_sent):
        batch_size = ar_sent.size(0)
        b_range = torch.arange(batch_size)
        ar_lengths = (ar_sent != self.ar_pad).int().sum(1) - 1
        en_lengths = (en_sent != self.en_pad).int().sum(1) - 1
        ar_len, en_len, feature_table, action_scores, _, _ = self._action_scores(
            ar_sent, en_sent)

        alpha = self._forward_evaluation(ar_sent, en_sent, action_scores)

        max_lens = torch.max(ar_lengths, en_lengths).float()
        log_probs = alpha[b_range.to(self.device), ar_lengths, en_lengths]

        return log_probs.exp(), (log_probs / max_lens).exp()


class EditDistNeuralModelProgressive(NeuralEditDistBase):
    """Model for sequence generation."""
    def __init__(self, ar_vocab, en_vocab, device, directed=True,
                 hidden_dim=32, hidden_layers=2, attention_heads=4,
                 window=3,
                 encoder_decoder_attention=True, model_type="transformer",
                 start_symbol="<s>", end_symbol="</s>", pad_symbol="<pad>"):
        super(EditDistNeuralModelProgressive, self).__init__(
            ar_vocab, en_vocab, device, directed, table_type="vocab",
            model_type=model_type,
            encoder_decoder_attention=encoder_decoder_attention,
            hidden_dim=hidden_dim, hidden_layers=hidden_layers,
            attention_heads=attention_heads, window=window,
            start_symbol=start_symbol, end_symbol=end_symbol,
            pad_symbol=pad_symbol)

    def forward(self, ar_sent, en_sent):
        b_range = torch.arange(ar_sent.size(0))
        ar_lengths = (ar_sent != self.ar_pad).int().sum(1) - 1
        en_lengths = (en_sent != self.en_pad).int().sum(1) - 1
        (ar_len, en_len, feature_table, action_scores,
            insertion_logits, subs_logits) = self._action_scores(
                ar_sent, en_sent)

        alpha = self._forward_evaluation(ar_sent, en_sent, action_scores)
        beta, expected_counts = self._backward_evalatuion_and_expectation(
            ar_len, en_len, ar_sent, en_sent, alpha, action_scores)

        insertion_log_dist = (
            F.log_softmax(insertion_logits, dim=3)
            + alpha[:, :, 1:].unsqueeze(3))
        subs_log_dist = (
            F.log_softmax(subs_logits, dim=3)
            + alpha[:, 1:, 1:].unsqueeze(3))

        next_symbol_logprobs_sum = torch.cat(
            (insertion_log_dist, subs_log_dist), dim=1).logsumexp(1)
        next_symbol_logprobs = (
            next_symbol_logprobs_sum -
            next_symbol_logprobs_sum.logsumexp(2, keepdims=True))

        distorted_probs = self._alpha_distortion_penalty(
            ar_len, en_len, alpha)

        return (action_scores, torch.exp(expected_counts),
                alpha[b_range, ar_lengths, en_lengths],
                next_symbol_logprobs, distorted_probs)

    @torch.no_grad()
    def _scores_for_next_step(
            self, b_range, v, feature_table, log_ar_mask, alpha):
        """Predict scores of next symbol, given the decoding history.

        The decoding history is already in the feature table that contains also
        the most recently decoded symbol.

        Args:
            b_range: Technical thing: tensor 0..batch_size
            v: Position in the decoding.
            feature_table: Representation of symbol pairs.
            log_ar_mask: Position maks for the input in the log domain.
            alpha: Table with state probabilties.

        Returns:
            Logits for the next symbols.
        """

        insertion_scores = (
            F.log_softmax(
                self.insertion_proj(feature_table[b_range, :, v - 1:v]),
                dim=-1)
            + log_ar_mask.unsqueeze(2).unsqueeze(3)
            + alpha[:, :, v - 1:v].unsqueeze(3))

        subs_scores = (
            F.log_softmax(
                self.substitution_proj(feature_table[b_range, 1:, v - 1:v]),
                dim=-1)
            + log_ar_mask[:, 1:].unsqueeze(2).unsqueeze(3)
            + alpha[:, 1:, v - 1:v].unsqueeze(3))

        next_symb_scores = torch.cat(
            (insertion_scores, subs_scores), dim=1).logsumexp(1)

        return next_symb_scores

    @torch.no_grad()
    def _update_alpha_with_new_row(
            self, batch_size, b_range, v, alpha, action_scores, ar_sent,
            en_sent, ar_len):
        """Update the probabilities table for newly decoded characters.

        Here, we add a row to the alpha table (probabilities of edit states)
        for the recently added symbol during decoding. It assumes the feature
        table already contains representations for the most recent symbol.

        Args:
            b_range: Technical thing: tensor 0..batch_size
            v: Position in the decoding.
            alpha: Alpha table from previous step.
            action_scores: Table with scores for particular edit actions.
            ar_sent: Source sequence.
            en_sent: Prefix of so far decoded target sequence.
            ar_len: Max lenght of the source sequences.

        Return:
            Updated alpha table.
        """
        alpha = torch.cat(
            (alpha, torch.full(
                (batch_size, ar_len, 1),  MINF).to(self.device)), dim=2)

        # TODO masking!
        for t, ar_char in enumerate(ar_sent.transpose(0, 1)):
            deletion_id = self._deletion_id(ar_char)
            insertion_id = self._insertion_id(en_sent[:, v])
            subsitute_id = self._substitute_id(ar_char, en_sent[:, v])

            to_sum = [
                action_scores[b_range, t, v, insertion_id] +
                alpha[:, t, v - 1]]
            if t >= 1:
                to_sum.append(
                    action_scores[b_range, t, v, deletion_id] +
                    alpha[:, t - 1, v])
                to_sum.append(
                    action_scores[b_range, t, v, subsitute_id] +
                    alpha[:, t - 1, v - 1])

            if len(to_sum) == 1:
                alpha[:, t, v] = to_sum[0]
            else:
                alpha[:, t, v] = torch.stack(to_sum).logsumexp(0)

        return alpha

    @torch.no_grad()
    def decode(self, ar_sent):
        batch_size = ar_sent.size(0)
        b_range = torch.arange(batch_size)

        en_sent = torch.tensor([[self.en_bos]] * batch_size).to(self.device)
        (ar_len, en_len, feature_table,
         action_scores, _, _) = self._action_scores(
            ar_sent, en_sent, inference=True)
        log_ar_mask = torch.where(
            ar_sent == self.ar_pad,
            torch.full_like(ar_sent, MINF, dtype=torch.float),
            torch.zeros_like(ar_sent, dtype=torch.float))

        # special case, v = 0
        alpha = torch.full((batch_size, ar_len, 1), MINF).to(self.device)
        for t, ar_char in enumerate(ar_sent.transpose(0, 1)):
            if t == 0:
                alpha[:, :, 0] = log_ar_mask
                continue
            deletion_id = self._deletion_id(ar_char)
            alpha[:, t, 0] = (
                action_scores[b_range, t, 0, deletion_id] + alpha[:, t - 1, 0])

        finished = torch.full(
            [batch_size], False, dtype=torch.bool).to(self.device)
        for v in range(1, 2 * ar_sent.size(1)):

            next_symb_scores = self._scores_for_next_step(
                b_range, v, feature_table, log_ar_mask, alpha)

            next_symbol_candidate = next_symb_scores.argmax(2)
            next_symbol = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_symbol_candidate, self.en_pad),
                next_symbol_candidate)

            en_sent = torch.cat((en_sent, next_symbol), dim=1)

            ar_len, en_len, feature_table, action_scores, _, _ = self._action_scores(
                ar_sent, en_sent, inference=True)
            finished += next_symbol.squeeze(1) == self.en_eos

            alpha = self. _update_alpha_with_new_row(
                batch_size, b_range, v, alpha, action_scores, ar_sent, en_sent, ar_len)

            # expand the target sequence
            if torch.all(finished):
                break

        return en_sent

    @torch.no_grad()
    def beam_search(self, ar_sent, beam_size=10):
        batch_size = ar_sent.size(0)
        b_range = torch.arange(batch_size)

        log_ar_mask = torch.where(
            ar_sent == self.ar_pad,
            torch.full_like(ar_sent, MINF, dtype=torch.float),
            torch.zeros_like(ar_sent, dtype=torch.float))

        # special case, v = 0 - intitialize first row of alpha table
        decoded = torch.full(
            (batch_size, 1, 1), self.en_bos, dtype=torch.long).to(self.device)
        (ar_len, en_len, feature_table,
         action_scores, _, _) = self._action_scores(
            ar_sent, decoded.squeeze(1), inference=True)

        alpha = torch.full((batch_size, 1, ar_len, 1), MINF).to(self.device)
        for t, ar_char in enumerate(ar_sent.transpose(0, 1)):
            if t == 0:
                alpha[:, 0, :, 0] = log_ar_mask
                continue
            deletion_id = self._deletion_id(ar_char)
            alpha[:, 0, t, 0] = (
                action_scores[b_range, t, 0, deletion_id] + alpha[:, 0, t - 1, 0])

        # INITIALIZE THE BEAM SEARCH
        cur_len = 1
        current_beam = 1
        finished = torch.full(
            (batch_size, 1, 1), False, dtype=torch.bool).to(self.device)
        scores = torch.zeros((batch_size, 1)).to(self.device)

        flat_decoded = decoded.reshape(batch_size, cur_len)
        flat_finished = finished.reshape(batch_size, cur_len)
        flat_alpha = alpha.reshape(batch_size, ar_len, 1)
        while cur_len < 2 * ar_len:
            next_symb_scores = self._scores_for_next_step(
                b_range, cur_len, feature_table, log_ar_mask, flat_alpha)

            # get scores of all expanded hypotheses
            candidate_scores = (
                scores.unsqueeze(2) +
                next_symb_scores.reshape(batch_size, current_beam, -1))

            # reshape for beam members and get top k
            best_scores, best_indices = candidate_scores.reshape(
                batch_size, -1).topk(beam_size, dim=-1)
            next_symbol_ids = best_indices % self.en_symbol_count
            hypothesis_ids = best_indices // self.en_symbol_count

            # numbering elements in the extended batch, i.e. beam size copies
            # of each batch element
            beam_offset = torch.arange(
                0, batch_size * current_beam, step=current_beam,
                dtype=torch.long, device=self.device)
            global_best_indices = (
                beam_offset.unsqueeze(1) + hypothesis_ids).reshape(-1)

            # now select appropriate histories
            decoded = torch.cat((
                flat_decoded.index_select(
                    0, global_best_indices).reshape(batch_size, beam_size, -1),
                next_symbol_ids.unsqueeze(-1)), dim=2)
            reordered_finished = flat_finished.index_select(
                0, global_best_indices)
            finished_now = (
                next_symbol_ids.view(-1, 1) == self.en_eos
                + reordered_finished[:, -1:])
            finished = torch.cat((
                reordered_finished,
                finished_now), dim=1).reshape(batch_size, beam_size, -1)
            flat_alpha = flat_alpha.index_select(0, global_best_indices)

            # re-order scores
            scores = best_scores
            # TODO need to be done better fi we want lenght normalization

            # tile encoder after first step
            if cur_len == 1:
                ar_sent = ar_sent.unsqueeze(1).repeat(
                    1, beam_size, 1).reshape(batch_size * beam_size, -1)
                log_ar_mask = log_ar_mask.unsqueeze(1).repeat(
                    1, beam_size, 1).reshape(batch_size * beam_size, -1)
                b_range = torch.arange(batch_size * beam_size).to(self.device)

            # prepare feature and alpha for the next step
            flat_decoded = decoded.reshape(-1, cur_len + 1)
            flat_finished = finished.reshape(-1, cur_len + 1)
            (ar_len, en_len, feature_table,
                action_scores, _, _) = self._action_scores(
                    ar_sent, flat_decoded, inference=True)
            flat_alpha = self. _update_alpha_with_new_row(
                flat_decoded.size(0), b_range, cur_len, flat_alpha,
                action_scores, ar_sent, flat_decoded, ar_len)

            # in the first iteration, beam size is 1, in the later ones,
            # it is the real beam size
            current_beam = beam_size
            cur_len += 1

        return decoded[:, 0]

    @torch.no_grad()
    def _viterbi_with_actions(self, ar_sent, en_sent, actions_scores):
        ar_len, en_len = ar_sent.size(1), en_sent.size(1)
        action_count = torch.zeros((ar_len, en_len))
        alpha = torch.zeros((ar_len, en_len)) + MINF

        alpha = torch.full((batch_size, ar_len, en_len), MINF)
        alpha[:, 0, 0] = 0
        for t, ar_char in enumerate(ar_sent.transpose(0, 1)):
            for v, en_char in enumerate(en_sent.transpose(0, 1)):
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


class EditDistStatModel(EditDistBase):
    """The original statistical algorithm by Ristad and Yanilos.

    This is a reimplemntation of the original learnable edit distance algorithm
    in PyTorch using the same interface as the neural models.
    """
    def __init__(self, ar_vocab, en_vocab, start_symbol="<s>",
                 end_symbol="</s>", pad_symbol="<pad>",
                 identitiy_initialize=True):
        super(EditDistStatModel, self).__init__(
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

    def forward(self, ar_sent, en_sent):
        ar_len, en_len = ar_sent.size(1), en_sent.size(1)
        table_shape = ((ar_len, en_len, self.n_target_classes))

        plausible_deletions = torch.zeros(table_shape) + MINF
        plausible_insertions = torch.zeros(table_shape) + MINF
        plausible_substitutions = torch.zeros(table_shape) + MINF

        alpha = torch.zeros((ar_len, en_len)) + MINF
        alpha[0, 0] = 0
        for t, ar_char in enumerate(ar_sent[0]):
            for v, en_char in enumerate(en_sent[0]):
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

        for t in reversed(range(ar_len)):
            for v in reversed(range(en_len)):

                to_sum = [beta[t, v]]
                if v < en_len - 1:
                    insertion_id = self._insertion_id(en_sent[0, v + 1])
                    to_sum.append(
                        self.weights[insertion_id] + beta[t, v + 1])
                if t < ar_len - 1:
                    deletion_id = self._deletion_id(ar_sent[0, t + 1])
                    to_sum.append(
                        self.weights[deletion_id] + beta[t + 1, v])
                if v < en_len - 1 and t < ar_len - 1:
                    subsitute_id = self._substitute_id(
                        ar_sent[0, t + 1], en_sent[0, v + 1])
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

    def maximize_expectation(self, expectations):
        epsilon = torch.log(torch.tensor(1e-16)) + \
            torch.zeros_like(self.weights)
        expecation_sum = torch.stack([epsilon] + expectations).logsumexp(0)
        distribution = (
            expecation_sum - expecation_sum.logsumexp(0, keepdim=True))

        self.weights = torch.stack([
            torch.log(torch.tensor(0.9)) + self.weights,
            torch.log(torch.tensor(0.1)) + distribution]).logsumexp(0)
