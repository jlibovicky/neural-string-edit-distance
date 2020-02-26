#!/usr/bin/env python3

import argparse

import torch
from torch import nn, optim
from torch.functional import F
from torchtext import data

from models import EditDistNeuralModelConcurrent


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("data_prefix", type=str)
    args = parser.parse_args()

    ar_text_field = data.Field(
        tokenize=list, init_token="<s>", eos_token="</s>", batch_first=True)
    en_text_field = data.Field(
        tokenize=list, init_token="<s>", eos_token="</s>", batch_first=True)
    labels_field = data.Field(sequential=False)

    train_data, val_data, test_data = data.TabularDataset.splits(
        path=args.data_prefix, train='train.txt',
        validation='eval.txt', test='test.txt', format='tsv',
        fields=[('ar', ar_text_field), ('en', en_text_field),
                ('labels', labels_field)])

    ar_text_field.build_vocab(train_data)
    en_text_field.build_vocab(train_data)
    labels_field.build_vocab(train_data)
    true_class_label = labels_field.vocab.stoi['1']

    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train_data, val_data, test_data), batch_sizes=(1, 1, 1),
        shuffle=True, device=0, sort_key=lambda x: len(x.ar))

    neural_model = EditDistNeuralModelConcurrent(
        ar_text_field.vocab, en_text_field.vocab)

    kl_div_loss = nn.KLDivLoss(reduction='batchmean')
    xent_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(neural_model.parameters())

    pos_examples = 0
    for i, train_ex in enumerate(train_iter):
        label = 1 if train_ex.labels[0] == true_class_label else -1

        pos_examples += 1

        action_scores, expected_counts, logprob = neural_model(
            train_ex.ar, train_ex.en)

        loss = -label * logprob
        if label == 1:
            #fake_targets = expected_counts.argmax(-1)
            loss += kl_div_loss(
                action_scores.reshape(-1, 4),
                expected_counts.reshape(-1, 4))
        loss += -label * xent_loss(
            action_scores.reshape(-1, 4),
            torch.full(action_scores.shape[:-1], 3, dtype=torch.long).reshape(-1))

        loss.backward()

        if pos_examples % 50 == 49:
            print(f"train loss = {loss.cpu():.10g}")
            optimizer.step()
            optimizer.zero_grad()

        if pos_examples % 50 == 49:
            neural_model.eval()
            with torch.no_grad():
                neural_false_scores = []
                neural_true_scores = []
                for j, val_ex in enumerate(val_iter):
                    neural_score = neural_model.viterbi(val_ex.ar, val_ex.en)
                    if j < 50:
                        if val_ex.labels == true_class_label:
                            neural_true_scores.append(neural_score)
                        else:
                            neural_false_scores.append(neural_score)
                    else:
                        print(f"neural true  scores: {sum(neural_true_scores) / len(neural_true_scores)}")
                        print(f"neural false scores: {sum(neural_false_scores) / len(neural_false_scores)}")
                        break
            neural_model.train()


if __name__ == "__main__":
    main()
