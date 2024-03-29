"""CNN (and embeddings) encoder and with the same API as Transoformers."""

import torch
from torch import nn
from torch.functional import F
from transformers.modeling_bert import BertSelfAttention

from rnn import AttConfig


class CNNEncoder(nn.Module):
    """CNN/Embeddings sequence encoder.

    With zero layers, only uses the input symbol embeddings with learned
    position embeddings.
    """
    def __init__(self, vocab, hidden_size, embedding_size, layers=0,
                 window=3, dropout=0.1):
        super().__init__()

        if window % 2 == 0:
            raise ValueError("Only even window sizes are allowed.")

        self.layers = layers
        self.embeddings = nn.Embedding(len(vocab), embedding_size)
        self.pos_embeddings = nn.Embedding(512, embedding_size)
        self.dropout = nn.Dropout(dropout)

        self.embedd_norm = nn.LayerNorm(embedding_size)

        if layers > 0:
            self.cnn_layers = nn.ModuleList()
            self.cnn_norms = nn.ModuleList()

            for _ in range(layers):
                self.cnn_layers.append(
                    nn.Conv1d(embedding_size, 2 * hidden_size,
                              window, padding=window - 1))
                self.cnn_norms.append(nn.LayerNorm(hidden_size))

    def forward(self, input_ids, attention_mask):
        input_range = torch.arange(
            input_ids.size(1)).unsqueeze(0).to(input_ids.device)

        output = (
            self.embeddings(input_ids) + self.pos_embeddings(input_range))
        output = self.embedd_norm(self.dropout(output))

        # Backward compatibility of saved models
        if hasattr(self, "layers"):
            for i in range(self.layers):
                cnn_output = output * attention_mask.float().unsqueeze(2)
                cnn_output = F.glu(self.cnn_layers[i](
                    output.transpose(2, 1)).transpose(
                        2, 1))[:, :input_ids.size(1)]
                output = self.cnn_norms[i](self.dropout(cnn_output) + output)
        else:
            if self.cnn_layer is not None:
                output = output * attention_mask.float().unsqueeze(2)
                output = self.cnn_layer(output.transpose(2, 1)).transpose(2, 1)
                output = self.cnn_norm(self.dropout(output))

        return output, None


class CNNDecoder(nn.Module):
    """CNN/Embeddings sequence encoder.

    With zero layers, only uses the input symbol embeddings with learned
    position embeddings. It theory, it can do encoder-decoder attention
    when only sing the embddings.
    """
    def __init__(self, vocab, hidden_size, embedding_size, layers=0, window=3,
                 use_attention=False, attention_heads=8, dropout=0.1,
                 output_proj=False):
        super().__init__()

        if window % 2 == 0:
            raise ValueError("Only even window sizes are allowed.")
        if output_proj:
            raise NotImplementedError()

        self.embeddings = nn.Embedding(len(vocab), embedding_size)
        self.pos_embeddings = nn.Embedding(512, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.use_attention = use_attention
        self.layers = layers

        self.embedd_norm = nn.LayerNorm(embedding_size)

        if layers > 0:
            self.cnn_layers = nn.ModuleList()
            self.cnn_norms = nn.ModuleList()
            self.atts = nn.ModuleList()
            self.att_norms = nn.ModuleList()

            for _ in range(layers):

                self.cnn_layers.append(
                    nn.Conv1d(embedding_size, 2 * hidden_size,
                              window, padding=window - 1))
                self.cnn_norms.append(nn.LayerNorm(hidden_size))

                if use_attention:
                    self.atts.append(BertSelfAttention(
                        AttConfig(hidden_size, attention_heads, True, dropout)))
                    self.att_norms.append(nn.LayerNorm(hidden_size))

    def forward(self, input_ids, attention_mask,
                encoder_hidden_states=None, encoder_attention_mask=None):
        input_range = torch.arange(
            input_ids.size(1)).unsqueeze(0).to(input_ids.device)

        output = (
            self.embeddings(input_ids) + self.pos_embeddings(input_range))
        output = self.embedd_norm(self.dropout(output))

        attentions = []
        if hasattr(self, "layers"):
            for i in range(self.layers):
                cnn_output = output * attention_mask.float().unsqueeze(2)
                cnn_output = F.glu(self.cnn_layers[i](
                    output.transpose(2, 1)).transpose(
                        2, 1))[:, :input_ids.size(1)]
                output = self.cnn_norms[i](self.dropout(cnn_output) + output)

                if self.use_attention and encoder_hidden_states is not None:
                    unsq_enc_att_mask = (
                        encoder_attention_mask.unsqueeze(1).unsqueeze(1))
                    context, att_dist = self.atts[i](
                        output, encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=unsq_enc_att_mask)
                    attentions.append(att_dist)
                    output = self.att_norms[i](output + self.dropout(context))
        else:
            if self.cnn_layer is not None:
                cnn_output = output * attention_mask.float().unsqueeze(2)
                cnn_output = self.cnn_layer(
                    output.transpose(2, 1)).transpose(
                        2, 1)[:, :input_ids.size(1)]
                output = self.cnn_norm(cnn_output + output)

            if encoder_hidden_states is not None:
                unsq_enc_att_mask = (
                    encoder_attention_mask.unsqueeze(1).unsqueeze(1))
                context = self.att(
                    output, encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=unsq_enc_att_mask)[0]
                output = self.att_norm(output + self.dropout(context))

        return output, None, attentions
