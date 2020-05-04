#!/usr/bin/env python3

import argparse

import torch
from torch import nn, optim
from torch.functional import F
from torchtext import data

from models import EditDistStatModel
from transliteration_utils import decode_ids


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("data_prefix", type=str)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    ar_text_field = data.Field(
        tokenize=list, init_token="<s>", eos_token="</s>", batch_first=True)
    en_text_field = data.Field(
        tokenize=list, init_token="<s>", eos_token="</s>", batch_first=True)

    train_data, val_data, test_data = data.TabularDataset.splits(
        path=args.data_prefix, train='train.txt',
        validation='eval.txt', test='test.txt', format='tsv',
        fields=[('ar', ar_text_field), ('en', en_text_field)])

    ar_text_field.build_vocab(train_data)
    en_text_field.build_vocab(train_data)

    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train_data, val_data, test_data), batch_sizes=(1, 1, 1),
        shuffle=True, device=0, sort_key=lambda x: len(x.ar))

    stat_model = EditDistStatModel(ar_text_field.vocab, en_text_field.vocab)

    pos_examples = 0
    stat_expecations = []
    for _ in range(args.epochs):
        for train_ex in train_iter:
            pos_examples += 1
            exp_counts = stat_model(train_ex.ar, train_ex.en)
            stat_expecations.append(exp_counts)

            if pos_examples % 1000 == 999:
                stat_model.maximize_expectation(stat_expecations)
                entropy = -(stat_model.weights * stat_model.weights.exp()).sum()
                print(f"stat. model entropy = {entropy.cpu():.10g}")
                stat_expecations = []

            if pos_examples % 1000 == 999:
                with torch.no_grad():
                    for j, val_ex in enumerate(val_iter):
                        stat_score = stat_model.viterbi(val_ex.ar, val_ex.en)
                        src_string = decode_ids(
                            val_ex.ar[0], ar_text_field)
                        tgt_string = decode_ids(
                            val_ex.en[0], en_text_field)

                        print(f"{src_string} -> {tgt_string}   {stat_score}")

                        if j > 10:
                            print()
                            break

    torch.save(stat_model, "stat_model.pt")


if __name__ == "__main__":
    main()
