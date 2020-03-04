#!/usr/bin/env python3

import argparse

import torch
from torch import nn, optim
from torch.functional import F
from torchtext import data

from models import EditDistStatModel


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

    stat_model = EditDistStatModel(ar_text_field.vocab, en_text_field.vocab)

    pos_examples = 0
    stat_expecations = []
    for i, train_ex in enumerate(train_iter):
        label = train_ex.labels[0] == true_class_label

        if not label:
            continue

        pos_examples += 1
        exp_counts = stat_model(train_ex.ar, train_ex.en)
        stat_expecations.append(exp_counts)

        if pos_examples % 50 == 49:
            stat_model.maximize_expectation(stat_expecations)
            entropy = -(stat_model.weights * stat_model.weights.exp()).sum()
            print(f"stat. model entropy = {entropy.cpu():.10g}")
            stat_expecations = []

        if pos_examples % 50 == 49:
            with torch.no_grad():
                stat_false_scores = []
                stat_true_scores = []
                for j, val_ex in enumerate(val_iter):
                    stat_score = stat_model.viterbi(val_ex.ar, val_ex.en)
                    if j < 50:
                        if val_ex.labels == true_class_label:
                            stat_true_scores.append(stat_score)
                        else:
                            stat_false_scores.append(stat_score)
                    else:
                        print(f"stat true  scores:   {sum(stat_true_scores) / len(stat_true_scores)}")
                        print(f"stat false scores:   {sum(stat_false_scores) / len(stat_false_scores)}")
                        break


if __name__ == "__main__":
    main()