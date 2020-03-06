import torch
from torch import nn, optim
from torch.functional import F

class RNNEncoder(nn.Module):
    def __init__(self, vocab, pad_token, hidden_size, embedding_size,
                 embedding, num_layers=2, dropout=0.0):
        super(RNNEncoder, self).__init__()

        self.pad_token = pad_token
        # Basic network params
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Embedding layer that will be shared with Decoder
        self.embedding = nn.Embedding(len(vocab), embedding_size)

        # Bidirectional GRU
        self.gru = nn.GRU(embedding_size, hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          batch_first=True,
                          bidirectional=True)

    def forward(self, input_sequence):
        input_lengths = (input_sequence != self.pad_token).sum(1)

        word_embeddings = self.embedding(input_sequence)
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            word_embeddings, input_lengths, batch_first=True, enforce_sorted=False)

        # Run the packed embeddings through the GRU, and then unpack the sequences
        outputs, hidden = self.gru(packed_embeddings)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        return outputs


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size


    def dot_score(self, hidden_state, encoder_states):
        return (hidden_state.unsqueeze(2) * encoder_states.unsqueeze(1)).sum(3)

    def forward(self, hidden, encoder_outputs, mask):
        attn_scores = self.dot_score(hidden, encoder_outputs)
        # Apply mask so network does not attend <pad> tokens
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, -1e10)

        attention_weights = F.softmax(attn_scores, dim=2).unsqueeze(3)

        context = (encoder_outputs.unsqueeze(1) * attention_weights).sum(2)
        return context


class RNNDecoder(nn.Module):
    def __init__(self, vocab, pad_token, hidden_size, embedding_size,
                 embedding, num_layers=2, dropout=0.0):
        super(RNNDecoder, self).__init__()


        self.pad_token = pad_token
        # Basic network params
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Embedding layer that will be shared with Decoder
        self.embedding = nn.Embedding(len(vocab), embedding_size)

        # Bidirectional GRU
        self.first_gru = nn.GRU(embedding_size, hidden_size,
                                num_layers=num_layers,
                                dropout=dropout,
                                batch_first=True)

        self.attn = nn.ModuleList([
            Attention(hidden_size) for _ in range(num_layers)])

        self.other_grus = nn.ModuleList([
            nn.GRU(hidden_size, hidden_size, dropout=dropout, batch_first=True)
            for _ in range(num_layers - 1)])

    def forward(self, input_sequence, encoder_outputs=None, encoder_mask=None):
        input_lengths = (input_sequence != self.pad_token).sum(1)

        word_embeddings = self.embedding(input_sequence)
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            word_embeddings, input_lengths, batch_first=True, enforce_sorted=False)

        # Run the packed embeddings through the GRU, and then unpack the sequences
        outputs, _ = self.first_gru(packed_embeddings)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        if encoder_outputs is not None:
            context = self.attn[0](outputs, encoder_outputs, encoder_mask)
            outputs = outputs + context

        for gru, att in zip(self.other_grus, self.attn[1:]):
            packed_outputs = nn.utils.rnn.pack_padded_sequence(
                word_embeddings, input_lengths, batch_first=True,
                enforce_sorted=False)
            next_outputs, _ = gru(packed_outputs)
            next_outputs, _ = nn.utils.rnn.pad_packed_sequence(next_outputs, batch_first=True)
            outputs = outputs + next_outputs

            if encoder_outputs is not None:
                context = self.attn[0](outputs, encoder_outputs, encoder_mask)
                outputs = outputs + context

        return outputs
