"""CNN (and embeddings) encoder and with the same API as Transoformers."""

import torch
from torch import nn
from torch.functional import F


class CNNEncoder(nn.Module):
    def __init__(self, vocab, hidden_size, embedding_size, layers=0, window=3):
        super(Encoder, self).__init__()

        if window % 2 == 0:
            raise ValueError("Only even window sizes are allowed.")

        self.embeddings = nn.Embedding(len(vocab), embedding_size)
        self.pos_embeddings = nn.Embedding(512, embedding_size)
        self.dropout = nn.Dropout(dropout)

        self.embedd_norm = nn.LayerNorm(embedding_size)

        self.cnn_layer = None
        if layers > 0:
            self.cnn_layer = nn.Conv1d(
                embedding_size, hidden_size, window, padding=(window - 2) / 2)
            self.cnn_norm = nn.LayerNorm(hidden_size)


    def forward(self, input_sequence, attention_mask):
        input_range = torch.arange(input_sequence.size(1)).unsqueeze(0)

        output = (
            self.embeddings(input_sequence) + self.pos_embeddings(input_range))
        output = self.embedd_norm(self.dropout(output))

        if cnn_layer is not None:
            output = self.cnn_layer(output.transpose(0, 2 ,1)).transpose(0, 2, 1)
            output = self.cnn_norm(self.dropout(output))

        return output


class CNNDecoder(nn.Module):
    def __init__(self, vocab, hidden_size, embedding_size, layers=0, window=3,
                 use_attention=False, attention_heads=8):
        super(Decoder, self).__init__()

        if window % 2 == 0:
            raise ValueError("Only even window sizes are allowed.")

        self.embeddings = nn.Embedding(len(vocab), embedding_size)
        self.pos_embeddings = nn.Embedding(512, embedding_size)
        self.dropout = nn.Dropout(dropout)

        self.embedd_norm = nn.LayerNorm(embedding_size)

        self.cnn_layer = None
        if layers > 0:
            self.cnn_layer = nn.Conv1d(
                embedding_size, hidden_size, window, padding=window -1)
            self.cnn_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, attention_mask,
                encoder_hidden_states=None, encoder_attention_mask=None):
        input_lengths = attention_mask.sum(1)

        input_range = torch.arange(input_sequence.size(1)).unsqueeze(0)

        output = (
            self.embeddings(input_sequence) + self.pos_embeddings(input_range))
        output = self.embedd_norm(self.dropout(output))

        if cnn_layer is not None:
            output = self.cnn_layer(output.transpose(0, 2 ,1)).transpose(0, 2, 1)[:, :-1]
            output = self.cnn_norm(self.dropout(output))

        return output
