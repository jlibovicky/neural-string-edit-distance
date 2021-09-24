"""Neural string edit distance."""

from typing import List, Tuple

import heapq
from collections import namedtuple

import torch
from torch import nn
from torch.functional import F
from torch import Tensor

from transformers import BertConfig, BertModel
from transformers.modeling_bert import BertSelfAttention

from rnn import RNNEncoder, RNNDecoder
from cnn import CNNEncoder, CNNDecoder

MINF = torch.log(torch.tensor(0.))


AttConfig = namedtuple(
    "AttConfig",
    ["hidden_size", "num_attention_heads", "output_attentions",
     "attention_probs_dropout_prob"])


class EditDistBase(nn.Module):
    """Base class used both for statistical and neural model."""
    def __init__(self, src_vocab, tgt_vocab, start_symbol,
                 end_symbol, pad_symbol, table_type="full", extra_classes=0):
        super().__init__()

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        self.src_bos = src_vocab[start_symbol]
        self.src_eos = src_vocab[end_symbol]
        self.src_pad = src_vocab[pad_symbol]
        self.tgt_bos = tgt_vocab[start_symbol]
        self.tgt_eos = tgt_vocab[end_symbol]
        self.tgt_pad = tgt_vocab[pad_symbol]

        self.extra_classes = extra_classes
        self.src_symbol_count = len(src_vocab)
        self.tgt_symbol_count = len(tgt_vocab)

        self.table_type = table_type

        if table_type == "full":
            self.deletion_classes = self.src_symbol_count
            self.insertion_classes = self.tgt_symbol_count
            self.subs_classes = self.src_symbol_count * self.tgt_symbol_count
            self.n_target_classes = (
                self.extra_classes +
                self.src_symbol_count +  # delete source
                self.tgt_symbol_count +  # insert target
                self.src_symbol_count * self.tgt_symbol_count)  # substitute
        elif table_type == "tiny":
            self.deletion_classes = 1
            self.insertion_classes = 1
            self.subs_classes = 1
            self.n_target_classes = self.extra_classes + 3
        elif table_type == "vocab":
            self.deletion_classes = 1
            self.insertion_classes = self.tgt_symbol_count
            self.subs_classes = self.tgt_symbol_count
            self.n_target_classes = (
                self.extra_classes + 1 + 2 * self.tgt_symbol_count)
        else:
            raise ValueError("Unknown table type.")

    def _deletion_id(self, src_char):
        if self.table_type == "full":
            return src_char
        return torch.zeros_like(src_char)

    def _insertion_id(self, tgt_char):
        if self.table_type == "full":
            return self.src_symbol_count + tgt_char
        if self.table_type == "tiny":
            return torch.ones_like(tgt_char)
        if self.table_type == "vocab":
            return 1 + tgt_char
        raise RuntimeError("Unknown table type.")

    def _substitute_id(self, src_char, tgt_char):
        if self.table_type == "full":
            subs_id = (self.src_symbol_count + self.tgt_symbol_count +
                       self.tgt_symbol_count * src_char + tgt_char)
            assert subs_id < self.n_target_classes
            return subs_id
        if self.table_type == "tiny":
            return torch.full_like(src_char, 2)
        if self.table_type == "vocab":
            return 1 + self.tgt_symbol_count + tgt_char
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
    def __init__(self, src_vocab, tgt_vocab, device, directed=False,
                 encoder_decoder_attention=True,
                 share_encoders=False,
                 table_type="vocab", extra_classes=0,
                 window=3,
                 hidden_dim=32, hidden_layers=2, attention_heads=4,
                 model_type="transformer",
                 start_symbol="<s>", end_symbol="</s>", pad_symbol="<pad>"):
        super().__init__(
            src_vocab, tgt_vocab, start_symbol, end_symbol, pad_symbol,
            table_type=table_type, extra_classes=extra_classes)

        if directed and share_encoders:
            raise ValueError(
                "You cannot share encoder if one of them is decoder.")
        if share_encoders:
            if src_vocab != tgt_vocab:
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
        self.src_encoder = self._encoder_for_vocab(src_vocab)
        if model_type == "bert" or share_encoders:
            self.tgt_encoder = self.src_encoder
        else:
            self.tgt_encoder = self._encoder_for_vocab(
                tgt_vocab, directed=directed)

        self.projection = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(self.hidden_dim))

        self.encoder_decoder_attention = encoder_decoder_attention

        proj_source = 4 * self.hidden_dim if self.directed else self.hidden_dim

        if self.directed:
            self.attention = BertSelfAttention(AttConfig(
                self.hidden_dim, 4, True, 0.1))

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
        if self.model_type == "rnn":
            return self._rnn_for_vocab(vocab, directed)
        if self.model_type == "bert":
            return BertModel.from_pretrained("bert-base-cased")
        if self.model_type == "embeddings":
            return self._cnn_for_vocab(vocab, directed, hidden=False)
        if self.model_type == "cnn":
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
        return CNNDecoder(
            vocab, self.hidden_dim,
            self.hidden_dim, layers=layers,
            window=self.window,
            attention_heads=self.attention_heads,
            output_proj=False, dropout=dropout)

    def _encode_src(self, inputs, mask):
        return self.src_encoder(inputs, attention_mask=mask)[0]

    def _encode_tgt(self, inputs, mask, src_vectors, src_mask):
        if self.encoder_decoder_attention:
            return self.tgt_encoder(
                inputs, attention_mask=mask,
                encoder_hidden_states=src_vectors,
                encoder_attention_mask=src_mask)[0]
        return self.tgt_encoder(
            inputs, attention_mask=mask)[0]

    def _target_class_ids(
            self, src_sent: Tensor, tgt_sent: Tensor):
        """Output classes for input string pair.

        This function computes what target classes correspond to deleting
        source symbols, inserting target symbols and subsitituting source
        symbols for target symbols. The vocabulary indices cannot be used
        directly because they are different on source and target side and there
        is a different number of possible edit operations than the vocabulary
        sizes.
        """
        all_deletion_ids = self._deletion_id(src_sent)
        all_insertion_ids = self._insertion_id(tgt_sent)
        all_subs_ids = self._substitute_id(
            src_sent.unsqueeze(2).repeat(1, 1, tgt_sent.size(1)),
            tgt_sent.unsqueeze(1).repeat(1, src_sent.size(1), 1))
        return (all_deletion_ids, all_insertion_ids, all_subs_ids)

    def _action_scores(self, src_sent, tgt_sent):
        """Compute possible action probabilities (Eq. 3 and 4)."""
        src_len, tgt_len = src_sent.size(1), tgt_sent.size(1)
        src_mask = src_sent != self.src_pad
        tgt_mask = tgt_sent != self.tgt_pad
        src_vectors = self._encode_src(src_sent, src_mask)
        tgt_vectors = self._encode_tgt(tgt_sent, tgt_mask, src_vectors, src_mask)

        # TODO: do the sort of residual connection I had in Munich
        feature_table = self.projection(torch.cat((
            src_vectors.unsqueeze(2).repeat(1, 1, tgt_len, 1),
            tgt_vectors.unsqueeze(1).repeat(1, src_len, 1, 1)), dim=3))

        if self.directed:
            unsq_enc_att_mask = (
                src_mask.unsqueeze(1).unsqueeze(1))
            att_output = self.attention(
                tgt_vectors, encoder_hidden_states=src_vectors,
                encoder_attention_mask=unsq_enc_att_mask)[0]

            feature_table = torch.cat(
                (feature_table,
                 src_vectors.unsqueeze(2).repeat(1, 1, tgt_len, 1),
                 tgt_vectors.unsqueeze(1).repeat(1, src_len, 1, 1),
                 att_output.unsqueeze(1).repeat(1, src_len, 1, 1)),
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

        assert action_scores.size(1) == src_len
        assert action_scores.size(2) == tgt_len
        assert action_scores.size(3) == self.n_target_classes

        return (src_len, tgt_len, feature_table, action_scores,
                valid_insertion_logits,
                valid_subs_logits)

    def _forward_evaluation(
            self,
            src_sent: Tensor,
            tgt_sent: Tensor,
            action_scores: Tensor,
            all_deletion_ids: Tensor = None,
            all_insertion_ids: Tensor = None,
            all_subs_ids: Tensor = None):
        """Differentiable forward pass through the model. Algorithm 1."""

        if all_deletion_ids is None:
            (all_deletion_ids,
             all_insertion_ids,
             all_subs_ids) = self._target_class_ids(src_sent, tgt_sent)

        return _torchscript_forward_evaluation(
            all_deletion_ids, all_insertion_ids, all_subs_ids,
            action_scores, self.device)

    @torch.no_grad()
    def _backward_evalatuion_and_expectation(
            self,
            src_len: int,
            tgt_len: int,
            src_sent: Tensor,
            tgt_sent: Tensor,
            all_deletion_ids: Tensor,
            all_insertion_ids: Tensor,
            all_subs_ids: Tensor,
            alpha: Tensor,
            action_scores: Tensor):
        """The backward pass through the edit distance table. Algorithm 2.

        Unlike, the forward pass it does not have to be differentiable, because
        it is only used to compute the expected distribution that is as a
        "target" in the EM loss.
        """

        src_lengths = (src_sent != self.src_pad).sum(1)
        tgt_lengths = (tgt_sent != self.tgt_pad).sum(1)

        return _torchscript_backward_evaluation(
            src_len, tgt_len,
            src_lengths, tgt_lengths,
            all_deletion_ids, all_insertion_ids, all_subs_ids,
            alpha, action_scores)

    def _alpha_distortion_penalty(self, src_len, tgt_len, alpha_table):
        """Penalty for the alphas being too high outside from the diagonal."""
        penalties = self.distortion_mask[:, :src_len, :tgt_len]
        return alpha_table.exp() * penalties

    @torch.no_grad()
    def viterbi(self, src_sent, tgt_sent):
        """Get a single best sequence of edit ops for a string pair."""
        assert src_sent.size(0) == 1
        src_len, tgt_len, _, action_scores, _, _ = self._action_scores(
            src_sent, tgt_sent)
        action_scores = action_scores.squeeze(0)

        action_count = torch.zeros((src_len, tgt_len))
        actions = torch.zeros((src_len, tgt_len))
        alpha = torch.zeros((src_len, tgt_len)) + MINF
        alpha[0, 0] = 0
        for t, src_char in enumerate(src_sent[0]):
            for v, tgt_char in enumerate(tgt_sent[0]):
                if t == 0 and v == 0:
                    continue

                deletion_id = self._deletion_id(src_char)
                insertion_id = self._insertion_id(tgt_char)
                subsitute_id = self._substitute_id(src_char, tgt_char)

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
                        ((action_scores[t, v, subsitute_id] +
                            alpha[t - 1, v - 1]),
                         action_count[t - 1, v - 1] + 1, 2))
                else:
                    possible_actions.append((-1e12, 1.0, 2))

                best_action_cost, best_action_count, best_action_id = max(
                    possible_actions, key=lambda x: x[0])

                alpha[t, v] = best_action_cost
                action_count[t, v] = best_action_count
                actions[t, v] = best_action_id

        operations = []
        t = src_len - 1
        v = tgt_len - 1
        while t > 0 or v > 0:
            if actions[t, v] == 1:
                operations.append(
                    ("delete", src_sent[0, t - 1].cpu().numpy(), t - 1))
                t -= 1
            elif actions[t, v] == 0:
                operations.append(
                    ("insert", tgt_sent[0, v].cpu().numpy(), v))
                v -= 1
            elif actions[t, v] == 2:
                operations.append(
                    ("subs",
                     (src_sent[0, t - 1].cpu().numpy(),
                      tgt_sent[0, v].cpu().numpy()),
                     (t - 1, v)))
                v -= 1
                t -= 1
        operations.reverse()

        return (torch.exp(alpha[-1, -1] / action_count[-1, -1]),
                operations)

    @torch.no_grad()
    def alpha(self, src_sent, tgt_sent):
        batch_size = src_sent.size(0)
        _, _, _, action_scores, _, _ = (
            self._action_scores(src_sent, tgt_sent))

        alphas = self._forward_evaluation(src_sent, tgt_sent, action_scores)

        return alphas


class EditDistNeuralModelConcurrent(NeuralEditDistBase):
    """Model for binary sequence-pari classification."""
    def __init__(self, src_vocab, tgt_vocab, device, directed=False,
                 hidden_dim=32, hidden_layers=2, attention_heads=4,
                 share_encoders=False,
                 model_type="transformer",
                 start_symbol="<s>", end_symbol="</s>", pad_symbol="<pad>"):
        super().__init__(
            src_vocab, tgt_vocab, device, directed,
            encoder_decoder_attention=False,
            share_encoders=share_encoders,
            hidden_dim=hidden_dim, hidden_layers=hidden_layers,
            attention_heads=attention_heads,
            table_type="tiny", extra_classes=1,
            start_symbol=start_symbol, end_symbol=end_symbol,
            pad_symbol=pad_symbol, model_type=model_type)

    def forward(self, src_sent, tgt_sent):
        batch_size = src_sent.size(0)
        b_range = torch.arange(batch_size)
        src_lengths = (src_sent != self.src_pad).int().sum(1) - 1
        tgt_lengths = (tgt_sent != self.tgt_pad).int().sum(1) - 1
        src_len, tgt_len, _, action_scores, _, _ = self._action_scores(
            src_sent, tgt_sent)

        (all_deletion_ids,
         all_insertion_ids,
         all_subs_ids) = self._target_class_ids(src_sent, tgt_sent)

        alpha = self._forward_evaluation(
            src_sent, tgt_sent, action_scores,
            all_deletion_ids, all_insertion_ids, all_subs_ids)
        _, expected_counts = self._backward_evalatuion_and_expectation(
            src_len, tgt_len, src_sent, tgt_sent,
            all_deletion_ids, all_insertion_ids, all_subs_ids,
            alpha, action_scores)

        distorted_probs = self._alpha_distortion_penalty(
            src_len, tgt_len, alpha)

        return (action_scores, torch.exp(expected_counts),
                alpha[b_range, src_lengths, tgt_lengths], distorted_probs)

    @torch.no_grad()
    def probabilities(self, src_sent, tgt_sent):
        batch_size = src_sent.size(0)
        b_range = torch.arange(batch_size)
        src_lengths = (src_sent != self.src_pad).int().sum(1) - 1
        tgt_lengths = (tgt_sent != self.tgt_pad).int().sum(1) - 1
        _, _, _, action_scores, _, _ = (
            self._action_scores(src_sent, tgt_sent))

        alpha = self._forward_evaluation(src_sent, tgt_sent, action_scores)

        max_lens = torch.max(src_lengths, tgt_lengths).float()
        log_probs = alpha[b_range.to(self.device), src_lengths, tgt_lengths]

        return log_probs.exp(), (log_probs / max_lens).exp()


class EditDistNeuralModelProgressive(NeuralEditDistBase):
    """Model for sequence generation."""
    def __init__(self, src_vocab, tgt_vocab, device, directed=True,
                 hidden_dim=32, hidden_layers=2, attention_heads=4,
                 window=3,
                 encoder_decoder_attention=True, model_type="transformer",
                 start_symbol="<s>", end_symbol="</s>", pad_symbol="<pad>"):
        super().__init__(
            src_vocab, tgt_vocab, device, directed, table_type="vocab",
            model_type=model_type,
            encoder_decoder_attention=encoder_decoder_attention,
            hidden_dim=hidden_dim, hidden_layers=hidden_layers,
            attention_heads=attention_heads, window=window,
            start_symbol=start_symbol, end_symbol=end_symbol,
            pad_symbol=pad_symbol)

    def forward(self, src_sent, tgt_sent):
        b_range = torch.arange(src_sent.size(0))
        src_lengths = (src_sent != self.src_pad).int().sum(1) - 1
        tgt_lengths = (tgt_sent != self.tgt_pad).int().sum(1) - 1
        (src_len, tgt_len, _, action_scores,
            insertion_logits, subs_logits) = self._action_scores(
                src_sent, tgt_sent)

        (all_deletion_ids,
         all_insertion_ids,
         all_subs_ids) = self._target_class_ids(src_sent, tgt_sent)

        alpha = self._forward_evaluation(
            src_sent, tgt_sent, action_scores,
            all_deletion_ids, all_insertion_ids, all_subs_ids)
        _, expected_counts = self._backward_evalatuion_and_expectation(
            src_len, tgt_len, src_sent, tgt_sent,
            all_deletion_ids, all_insertion_ids, all_subs_ids,
            alpha, action_scores)

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
            src_len, tgt_len, alpha)

        return (action_scores, torch.exp(expected_counts),
                alpha[b_range, src_lengths, tgt_lengths],
                next_symbol_logprobs, distorted_probs)

    @torch.no_grad()
    def _scores_for_next_step(
            self, b_range, v, feature_table, log_src_mask, alpha):
        """Predict scores of next symbol, given the decoding history.

        The decoding history is already in the feature table that contains also
        the most recently decoded symbol.

        Args:
            b_range: Technical thing: tensor 0..batch_size
            v: Position in the decoding.
            feature_table: Representation of symbol pairs.
            log_src_mask: Position maks for the input in the log domain.
            alpha: Table with state probabilties.

        Returns:
            Logits for the next symbols.
        """

        insertion_scores = (
            F.log_softmax(
                self.insertion_proj(feature_table[b_range, :, v - 1:v]),
                dim=-1)
            + log_src_mask.unsqueeze(2).unsqueeze(3)
            + alpha[:, :, v - 1:v].unsqueeze(3))

        subs_scores = (
            F.log_softmax(
                self.substitution_proj(feature_table[b_range, 1:, v - 1:v]),
                dim=-1)
            + log_src_mask[:, 1:].unsqueeze(2).unsqueeze(3)
            + alpha[:, 1:, v - 1:v].unsqueeze(3))

        next_symb_scores = torch.cat(
            (insertion_scores, subs_scores), dim=1).logsumexp(1)

        return next_symb_scores

    @torch.no_grad()
    def _update_alpha_with_new_row(
            self, batch_size, b_range, v, alpha, action_scores, src_sent,
            tgt_sent, src_len):
        """Update the probabilities table for newly decoded characters.

        Here, we add a row to the alpha table (probabilities of edit states)
        for the recently added symbol during decoding. It assumes the feature
        table already contains representations for the most recent symbol.

        Args:
            b_range: Technical thing: tensor 0..batch_size
            v: Position in the decoding.
            alpha: Alpha table from previous step.
            action_scores: Table with scores for particular edit actions.
            src_sent: Source sequence.
            tgt_sent: Prefix of so far decoded target sequence.
            src_len: Max lenght of the source sequences.

        Return:
            Updated alpha table.
        """
        alpha = torch.cat(
            (alpha, torch.full(
                (batch_size, src_len, 1),  MINF).to(self.device)), dim=2)

        # TODO masking!
        for t, src_char in enumerate(src_sent.transpose(0, 1)):
            deletion_id = self._deletion_id(src_char)
            insertion_id = self._insertion_id(tgt_sent[:, v])
            subsitute_id = self._substitute_id(src_char, tgt_sent[:, v])

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
    def decode(self, src_sent):
        batch_size = src_sent.size(0)
        b_range = torch.arange(batch_size)

        tgt_sent = torch.tensor([[self.tgt_bos]] * batch_size).to(self.device)
        (src_len, _, feature_table,
         action_scores, _, _) = self._action_scores(src_sent, tgt_sent)
        log_src_mask = torch.where(
            src_sent == self.src_pad,
            torch.full_like(src_sent, MINF, dtype=torch.float),
            torch.zeros_like(src_sent, dtype=torch.float))

        # special case, v = 0
        alpha = torch.full((batch_size, src_len, 1), MINF).to(self.device)
        for t, src_char in enumerate(src_sent.transpose(0, 1)):
            if t == 0:
                alpha[:, :, 0] = log_src_mask
                continue
            deletion_id = self._deletion_id(src_char)
            alpha[:, t, 0] = (
                action_scores[b_range, t, 0, deletion_id] + alpha[:, t - 1, 0])

        finished = torch.full(
            [batch_size], False, dtype=torch.bool).to(self.device)
        for v in range(1, 2 * src_sent.size(1)):

            next_symb_scores = self._scores_for_next_step(
                b_range, v, feature_table, log_src_mask, alpha)

            next_symbol_candidate = next_symb_scores.argmax(2)
            next_symbol = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_symbol_candidate, self.tgt_pad),
                next_symbol_candidate)

            tgt_sent = torch.cat((tgt_sent, next_symbol), dim=1)

            (src_len, tgt_len, feature_table,
                    action_scores, _, _) = self._action_scores(
                src_sent, tgt_sent)
            finished += next_symbol.squeeze(1) == self.tgt_eos

            alpha = self. _update_alpha_with_new_row(
                batch_size, b_range, v, alpha, action_scores,
                src_sent, tgt_sent, src_len)

            # expand the target sequence
            if torch.all(finished):
                break

        return tgt_sent

    @torch.no_grad()
    def beam_search(self, src_sent, beam_size=10, len_norm=1.0):
        batch_size = src_sent.size(0)
        b_range = torch.arange(batch_size)

        log_src_mask = torch.where(
            src_sent == self.src_pad,
            torch.full_like(src_sent, MINF, dtype=torch.float),
            torch.zeros_like(src_sent, dtype=torch.float))

        # special case, v = 0 - intitialize first row of alpha table
        decoded = torch.full(
            (batch_size, 1, 1), self.tgt_bos, dtype=torch.long).to(self.device)
        (src_len, _, feature_table,
         action_scores, _, _) = self._action_scores(
            src_sent, decoded.squeeze(1))

        alpha = torch.full((batch_size, 1, src_len, 1), MINF).to(self.device)
        for t, src_char in enumerate(src_sent.transpose(0, 1)):
            if t == 0:
                alpha[:, 0, :, 0] = log_src_mask
                continue
            deletion_id = self._deletion_id(src_char)
            alpha[:, 0, t, 0] = (
                action_scores[b_range, t, 0, deletion_id]
                + alpha[:, 0, t - 1, 0])

        # INITIALIZE THE BEAM SEARCH
        cur_len = 1
        current_beam = 1
        finished = torch.full(
            (batch_size, 1, 1), False, dtype=torch.bool).to(self.device)
        scores = torch.zeros((batch_size, 1)).to(self.device)

        flat_decoded = decoded.reshape(batch_size, cur_len)
        flat_finished = finished.reshape(batch_size, cur_len)
        flat_alpha = alpha.reshape(batch_size, src_len, 1)
        while cur_len < 2 * src_len:
            next_symb_scores = self._scores_for_next_step(
                b_range, cur_len, feature_table, log_src_mask, flat_alpha)

            # get scores of all expanded hypotheses
            candidate_scores = (
                scores.unsqueeze(2) +
                next_symb_scores.reshape(batch_size, current_beam, -1))
            norm_factor = torch.pow(
                (1 - finished.float()).sum(2, keepdim=True) + 1, len_norm)
            normed_scores = candidate_scores / norm_factor

            # reshape for beam members and get top k
            _, best_indices = normed_scores.reshape(
                batch_size, -1).topk(beam_size, dim=-1)
            next_symbol_ids = best_indices % self.tgt_symbol_count
            hypothesis_ids = best_indices // self.tgt_symbol_count

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
                (next_symbol_ids.view(-1, 1) == self.tgt_eos)
                + reordered_finished[:, -1:])
            finished = torch.cat((
                reordered_finished,
                finished_now), dim=1).reshape(batch_size, beam_size, -1)
            flat_alpha = flat_alpha.index_select(0, global_best_indices)

            if finished_now.all():
                break

            # re-order scores
            scores = candidate_scores.reshape(
                batch_size, -1).gather(-1, best_indices)

            # TODO need to be done better fi we want lenght normalization

            # tile encoder after first step
            if cur_len == 1:
                src_sent = src_sent.unsqueeze(1).repeat(
                    1, beam_size, 1).reshape(batch_size * beam_size, -1)
                log_src_mask = log_src_mask.unsqueeze(1).repeat(
                    1, beam_size, 1).reshape(batch_size * beam_size, -1)
                b_range = torch.arange(batch_size * beam_size).to(self.device)

            # prepare feature and alpha for the next step
            flat_decoded = decoded.reshape(-1, cur_len + 1)
            flat_finished = finished.reshape(-1, cur_len + 1)
            (src_len, _, feature_table,
                action_scores, _, _) = self._action_scores(
                    src_sent, flat_decoded)
            flat_alpha = self. _update_alpha_with_new_row(
                flat_decoded.size(0), b_range, cur_len, flat_alpha,
                action_scores, src_sent, flat_decoded, src_len)

            # in the first iteration, beam size is 1, in the later ones,
            # it is the real beam size
            current_beam = beam_size
            cur_len += 1

        return decoded[:, 0]

    @torch.no_grad()
    def operation_decoding(self, src_sent):
        """Decode sequeence by operation sampling.

        Instead of sampling from symbol distributions, it samples directly
        operations.

        Args:
            at_sent: Source sequence.

        Returns:
            Decoded target string.

        """
        if src_sent.size(0) != 1:
            raise ValueError("Only works with batch size 1.")

        tgt_sent = torch.tensor([[self.tgt_bos]]).to(self.device)
        (src_len, _, feature_table, _, _, _) = self._action_scores(
            src_sent, tgt_sent)

        v = 0
        t = 0
        for _ in range(1, 2 * src_sent.size(1)):
            state = feature_table[0, t, v]

            if t >= src_len - 1:
                deletion_score = MINF.unsqueeze(0).to(self.device)
            else:
                deletion_score = self.deletion_logit_proj(state)
            insertion_scores = self.insertion_proj(state)
            if t >= src_len - 1:
                subs_scores = torch.full_like(insertion_scores, MINF)
            else:
                subs_scores = self.substitution_proj(state)

            all_scores = torch.cat([
                deletion_score, insertion_scores, subs_scores], dim=0)
            best_operation = all_scores.argmax()

            # Delete source symbol and move in the source sequence
            if best_operation == 0:
                t += 1
                continue

            # We are adding a new symbol => increase target sequnece index
            next_symbol = (best_operation - 1) % self.tgt_symbol_count
            v += 1

            if best_operation > self.tgt_symbol_count:
                t += 1

            tgt_sent = torch.cat((
                tgt_sent, next_symbol.unsqueeze(0).unsqueeze(0)), dim=1)

            (src_len, _, feature_table, _, _, _) = self._action_scores(
                src_sent, tgt_sent)
            if next_symbol == self.tgt_eos:
                break
        return tgt_sent

    @torch.no_grad()
    def operation_beam_search(self, src_sent, beam_size):
        """Decode sequeence by operation sampling.

        Instead of sampling from symbol distributions, it samples directly
        operations.

        Args:
            at_sent: Source sequence.

        Returns:
            Decoded target string.
        """
        if src_sent.size(0) != 1:
            raise ValueError("Only works with batch size 1.")

        # State is a tuple: target sequence, t, v, finished, score
        beam = [(torch.tensor([[self.tgt_bos]]).to(self.device), 0, 0, False, 0)]
        next_candidates = []

        def score_fn(hypothesis):
            _, _, tgt_pos, _, score_sum = hypothesis
            return score_sum / (tgt_pos + 1)

        for _ in range(1, 2 * src_sent.size(1)):
            for tgt_sent, t, v, finished, score in beam:
                if finished:
                    next_candidates.append((tgt_sent, t, v, finished, score))
                    continue

                (src_len, _, feature_table, _, _, _) = self._action_scores(
                    src_sent, tgt_sent)
                state = feature_table[0, t, v]

                if t >= src_len - 1:
                    deletion_score = MINF.unsqueeze(0).to(self.device)
                else:
                    deletion_score = self.deletion_logit_proj(state)
                insertion_scores = self.insertion_proj(state)
                if t >= src_len - 1:
                    subs_scores = torch.full_like(insertion_scores, MINF)
                else:
                    subs_scores = self.substitution_proj(state)

                all_scores = F.log_softmax(torch.cat([
                    deletion_score, insertion_scores, subs_scores], dim=0),
                    dim=0)
                best_scores, best_indices = all_scores.topk(2 * beam_size)

                for op_score, op_idx in zip(best_scores, best_indices):
                    # Delete source symbol and move in the source sequence
                    if op_idx == 0:
                        next_candidates.append((
                            tgt_sent, t + 1, v, False, score + op_score))
                        continue

                    # Adding a new symbol => increase target sequnece index
                    next_symbol = (op_idx - 1) % self.tgt_symbol_count
                    new_tgt_sent = torch.cat((
                        tgt_sent, next_symbol.unsqueeze(0).unsqueeze(0)), dim=1)

                    next_candidates.append((
                        new_tgt_sent,
                        t + (op_idx > self.tgt_symbol_count),
                        v + 1,
                        next_symbol == self.tgt_eos,
                        score + op_score))

            beam = heapq.nlargest(
                beam_size, next_candidates, key=score_fn)
            next_candidates = []
            if all(hyp[3] for hyp in beam):
                break

        return max(beam, key=score_fn)[0]


@torch.jit.script
def _torchscript_forward_evaluation(
        all_deletion_ids: Tensor,
        all_insertion_ids: Tensor,
        all_subs_ids: Tensor,
        action_scores: Tensor,
        device: torch.device) -> Tensor:

    """Differentiable forward pass through the model. Algorithm 1.

    Here, the input and otput sequence is no longer represented by vocabulary
    indices (they are necessary to get input embeddings), but by indices that
    corresponding to target classes of the edit operations.
    """
    minf = torch.log(torch.tensor(0.))
    alpha: List[List[Tensor]] = []
    batch_size = all_deletion_ids.size(0)
    b_range = torch.arange(batch_size).to(device)
    for t, deletion_id in enumerate(all_deletion_ids.transpose(0, 1)):
        alpha.append([])
        for v, insertion_id in enumerate(all_insertion_ids.transpose(0, 1)):
            if t == 0 and v == 0:
                alpha[0].append(torch.zeros((batch_size,)).to(device))
                continue

            subsitute_id = all_subs_ids[:, t, v]

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
                    torch.full([batch_size], minf).to(device))
            if len(to_sum) == 1:
                alpha[t].append(to_sum[0])
            else:
                alpha[t].append(
                    torch.stack(to_sum).logsumexp(0))

    alpha_tensor = torch.stack(
        [torch.stack(v) for v in alpha]).permute(2, 0, 1)
    return alpha_tensor


@torch.jit.script
def _torchscript_backward_evaluation(
        src_len: int,
        tgt_len: int,
        src_lengths: Tensor,
        tgt_lengths: Tensor,
        all_deletion_ids: Tensor,
        all_insertion_ids: Tensor,
        all_subs_ids: Tensor,
        alpha: Tensor,
        action_scores: Tensor) -> Tuple[Tensor, Tensor]:
    """The backward pass through the edit distance table.

    Compilable as a TorchScript function.
    """

    minf = torch.log(torch.tensor(0.))
    plausible_deletions = torch.full_like(action_scores, minf)
    plausible_insertions = torch.full_like(action_scores, minf)
    plausible_substitutions = torch.full_like(action_scores, minf)

    batch_size = all_deletion_ids.size(0)
    b_range = torch.arange(batch_size)

    beta = torch.full_like(alpha, minf)
    beta[:, -1, -1] = 0.0

    for t in torch.arange(src_len).to(alpha.device).flip(0):
        for v in torch.arange(tgt_len).to(alpha.device).flip(0):
            # Bool mask: when we are in the table inside both words
            is_valid = (v <= (tgt_lengths - 1)) * (t <= (src_lengths - 1))
            # Bool mask: true for end state of word pairs
            is_corner = (v == (tgt_lengths - 1)) * (t == (src_lengths - 1))

            to_sum = [beta[:, t, v]]
            if v < tgt_len - 1:
                insertion_id = all_insertion_ids[:, v]
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
                    torch.full_like(insertion_score_candidate, minf)))
            else:
                # This is here, so that TorchScript compiler does not complain.
                insertion_score_candidate = torch.zeros([batch_size])
            if t < src_len - 1:
                deletion_id = all_deletion_ids[:, t]
                plausible_deletions[b_range, t, v, deletion_id] = 0

                deletion_score_candidate = (
                    action_scores[b_range, t, v, deletion_id] +
                    beta[:, t + 1, v])

                to_sum.append(torch.where(
                    is_valid,
                    deletion_score_candidate,
                    torch.full_like(deletion_score_candidate, minf)))
            if v < tgt_len - 1 and t < src_len - 1:
                subsitute_id = all_subs_ids[:, t, v]
                plausible_substitutions[
                    b_range, t, v, subsitute_id] = 0

                substitution_score_candidate = (
                    action_scores[b_range, t, v, subsitute_id] +
                    beta[:, t + 1, v + 1])

                to_sum.append(torch.where(
                    is_valid,
                    substitution_score_candidate,
                    torch.full_like(insertion_score_candidate, minf)))

            beta_candidate = torch.stack(to_sum).logsumexp(0)

            beta[:, t, v] = torch.where(
                is_corner, torch.zeros_like(beta_candidate),
                torch.where(is_valid, beta_candidate,
                            torch.full_like(beta_candidate, minf)))

    # deletion expectation
    expected_deletions = torch.zeros_like(action_scores) + minf
    expected_deletions[:, 1:, :] = (
        alpha[:, :-1, :].unsqueeze(3) +
        action_scores[:, 1:, :] + plausible_deletions[:, 1:, :] +
        beta[:, 1:, :].unsqueeze(3))
    # insertions expectation
    expected_insertions = torch.zeros_like(action_scores) + minf
    expected_insertions[:, :, 1:] = (
        alpha[:, :, :-1].unsqueeze(3) +
        action_scores[:, :, 1:] + plausible_insertions[:, :, 1:] +
        beta[:, :, 1:].unsqueeze(3))
    # substitution expectation
    expected_substitutions = torch.zeros_like(action_scores) + minf
    expected_substitutions[:, 1:, 1:] = (
        alpha[:, :-1, :-1].unsqueeze(3) +
        action_scores[:, 1:, 1:] + plausible_substitutions[:, 1:, 1:] +
        beta[:, 1:, 1:].unsqueeze(3))

    expected_counts = torch.stack([
        expected_deletions, expected_insertions,
        expected_substitutions], dim=4).logsumexp(4)
    expected_counts -= expected_counts.logsumexp(3, keepdim=True)
    return beta, expected_counts
