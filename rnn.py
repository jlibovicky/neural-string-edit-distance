"""RNN Encoder and Decoder with the same call API as Transformers."""


from collections import namedtuple

import torch
from torch import nn
from torch.functional import F

from transformers.modeling_bert import BertSelfAttention


ATT_CONFIG = namedtuple("AttConfig", ["hidden_size", "num_attention_heads", "output_attentions", "attention_probs_dropout_prob"])


class RNNEncoder(nn.Module):
    def __init__(self, vocab, hidden_size, embedding_size,
                 num_layers=2, dropout=0.0):
        super(RNNEncoder, self).__init__()

        # Basic network params
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)
        self.embeddings = nn.Embedding(len(vocab), embedding_size)

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)])

        self.first_gru = nn.GRU(embedding_size, hidden_size,
                                num_layers=1,
                                batch_first=True, bidirectional=True)

        self.other_grus = nn.ModuleList([
            nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
            for _ in range(num_layers - 1)])

    def forward(self, input_sequence, attention_mask):
        input_lengths = attention_mask.sum(1)

        word_embeddings = self.dropout(self.embeddings(input_sequence))
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            word_embeddings, input_lengths, batch_first=True, enforce_sorted=False)

        # Run the packed embeddings through the GRU, and then unpack the sequences
        outputs, _ = self.first_gru(packed_embeddings)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs = self.norms[0](self.dropout(
            outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]))

        for gru, norm in zip(self.other_grus, self.norms[1:]):
            packed_outputs = nn.utils.rnn.pack_padded_sequence(
                outputs, input_lengths, batch_first=True,
                enforce_sorted=False)
            next_outputs, _ = gru(packed_outputs)
            next_outputs, _ = nn.utils.rnn.pad_packed_sequence(next_outputs, batch_first=True)
            next_outputs = next_outputs[:, :, :self.hidden_size] + next_outputs[:, : ,self.hidden_size:]
            outputs = norm(outputs + self.dropout(next_outputs))

        return outputs, None


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size


    def dot_score(self, hidden_state, encoder_states):
        return (hidden_state.unsqueeze(2) * encoder_states.unsqueeze(1)).sum(3)

    def forward(self, hidden, keys, values, mask):
        attn_scores = self.dot_score(hidden, keys)
        # Apply mask so network does not attend <pad> tokens
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, -1e10)

        attention_weights = F.softmax(attn_scores, dim=2).unsqueeze(3)

        context = (values.unsqueeze(1) * attention_weights).sum(2)
        return context


class RNNDecoder(nn.Module):
    def __init__(self, vocab, hidden_size, embedding_size,
                 num_layers=2, attention_heads=4, dropout=0.0,
                 use_attention=True,
                 output_proj=False):
        super(RNNDecoder, self).__init__()

        # Basic network params
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.use_attention = use_attention

        self.dropout = nn.Dropout(dropout)
        self.embeddings = nn.Embedding(len(vocab), embedding_size)

        self.first_gru = nn.GRU(embedding_size, hidden_size,
                                num_layers=num_layers,
                                batch_first=True)

        self.rnn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.ctx_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)])

        self.attn = [None for _ in range(num_layers)]
        if use_attention and attention_heads == 1:
            self.attn = nn.ModuleList([
                Attention(hidden_size) for _ in range(num_layers)])
        elif use_attention:
            self.attn = nn.ModuleList([
                BertSelfAttention(ATT_CONFIG(hidden_size, attention_heads, False, 0.1))
                for _ in range(num_layers)])

        self.other_grus = nn.ModuleList([
            nn.GRU(hidden_size, hidden_size, batch_first=True)
            for _ in range(num_layers - 1)])

        self.output_proj = None
        if output_proj:
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout))

    def forward(self, input_ids, attention_mask, encoder_hidden_states=None, encoder_attention_mask=None):
        input_lengths = attention_mask.sum(1)

        word_embeddings = self.dropout(self.embeddings(input_ids))
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            word_embeddings, input_lengths, batch_first=True, enforce_sorted=False)

        # Run the packed embeddings through the GRU, and then unpack the sequences
        outputs, _ = self.first_gru(packed_embeddings)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs = self.rnn_norms[0](self.dropout(outputs))

        if encoder_hidden_states is not None:
            context = self.attn[0](
                    outputs, encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask.unsqueeze(1).unsqueeze(1))[0]
            outputs = self.ctx_norms[0](outputs + self.dropout(context))

        for gru, att, rnn_norm, ctx_norm in zip(
                self.other_grus, self.attn[1:], self.rnn_norms[1:], self.ctx_norms[1:]):
            packed_outputs = nn.utils.rnn.pack_padded_sequence(
                outputs, input_lengths, batch_first=True,
                enforce_sorted=False)
            next_outputs, _ = gru(packed_outputs)
            next_outputs, _ = nn.utils.rnn.pad_packed_sequence(next_outputs, batch_first=True)
            outputs = rnn_norm(outputs + self.dropout(next_outputs))

            if encoder_hidden_states is not None:
                context = att(
                    outputs, encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask.unsqueeze(1).unsqueeze(1))[0]
                outputs = ctx_norm(outputs + self.dropout(context))

        if self.output_proj is not None:
            outputs = self.output_proj(outputs)

        return outputs, None
