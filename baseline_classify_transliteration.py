#!/usr/bin/env python3


import argparse
from itertools import chain
import math

import torch
from torch import nn, optim
from torchtext import data


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

        output = (output * mask.unsqueeze(2)).sum(0) / mask.unsqueeze(2).sum(0)
        return output


class RNNEncoder(nn.Module):
    def __init__(self, vocab, n_layers, hidden_dim):
        super(RNNEncoder, self).__init__()

        self.vocab = vocab
        self.embeddings = nn.Embedding(len(vocab), hidden_dim)
        self.rnn = nn.LSTM(hidden_size, hidden_size, dropout=0.1, bidirectional=True)

    def forward(self, data):
        pass


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


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("data_prefix", type=str)
    args = parser.parse_args()

    src_text_field = data.Field(tokenize=list)
    tgt_text_field = data.Field(tokenize=list)
    labels_field = data.Field(sequential=False)

    train_data, val_data, test_data = data.TabularDataset.splits(
        path=args.data_prefix, train='train.txt',
        validation='eval.txt', test='test.txt', format='tsv',
        fields=[('ar', src_text_field), ('en', tgt_text_field),
                ('labels', labels_field)])

    src_text_field.build_vocab(train_data)
    tgt_text_field.build_vocab(train_data)
    labels_field.build_vocab(train_data)
    true_class_label = labels_field.vocab.stoi['1']
    import ipdb; ipdb.set_trace()

    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train_data, val_data, test_data), batch_sizes=(256, 256, 256),
        shuffle=True, device=0, sort_key=lambda x: len(x.ar))

    src_transformer = Encoder(src_text_field.vocab, 2, 128, 4)
    tgt_transformer = Encoder(tgt_text_field.vocab, 2, 128, 4)
    classifier = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(2 * 128, 128),
        nn.Dropout(0.1),
        nn.ReLU(),
        nn.Linear(128, 1))

    xent = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        chain(src_transformer.parameters(),
              tgt_transformer.parameters(),
              classifier.parameters()))

    for _ in range(10):
        for i, batch in enumerate(train_iter):
            src_vector = ar_transformer(batch.ar)
            tgt_vector = en_transformer(batch.en)

            output = classifier(torch.cat((src_vector, tgt_vector), 1))
            target = (batch.labels == true_class_label).float()

            loss = xent(output.squeeze(1), target)
            print(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 20 == 19:
                src_transformer.eval()
                tgt_transformer.eval()
                classifier.eval()
                with torch.no_grad():
                    correct_count = 0

                    for val_batch in val_iter:
                        src_vector = ar_transformer(val_batch.ar)
                        tgt_vector = en_transformer(val_batch.en)
                        output = classifier(torch.cat((src_vector, tgt_vector), 1))
                        prediction = output > 0.0
                        target = val_batch.labels == true_class_label
                        correct_count += (target == prediction.squeeze(1)).float().sum()

                    print(correct_count / len(val_data))

                src_transformer.train()
                tgt_transformer.train()
                classifier.train()


if __name__ == "__main__":
    main()
