#!/usr/bin/env python3

import argparse

import torch
from torch import nn, optim
from torch.functional import F
from torchtext import data

from models import EditDistStatModel, EditDistNeuralModel


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("data_prefix", type=str)
    parser.add_argument("--em-loss", default=None, type=float)
    parser.add_argument("--nll-loss", default=None, type=float)
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

    neural_model = EditDistNeuralModel(
        ar_text_field.vocab, en_text_field.vocab, directed=True)

    kl_div = nn.KLDivLoss(reduction='batchmean')
    nll = nn.NLLLoss()
    optimizer = optim.Adam(neural_model.parameters())

    en_examples = []
    ar_examples = []
    pos_examples = 0
    for _ in range(10):
        for i, train_ex in enumerate(train_iter):
            pos_examples += 1

            (action_scores, expected_counts, action_entropy,
                logprob, next_symbol_score) = neural_model(
                train_ex.ar, train_ex.en)

            loss = torch.tensor(0.)
            kl_loss = 0
            if args.em_loss is not None:
                kl_loss = kl_div(action_scores, expected_counts)
                loss += args.em_loss * kl_loss

            nll_loss = 0
            if args.nll_loss is not None:
                nll_loss = nll(next_symbol_score[:-1], train_ex.en[0, 1:])
                loss += args.nll_loss * nll_loss

            loss.backward()

            if pos_examples % 50 == 49:
                print(f"train loss = {loss:.3g} (NLL {nll_loss:.3g}, EM: {kl_loss:.3g})")
                optimizer.step()
                optimizer.zero_grad()

            if pos_examples % 500 == 499:
                neural_model.eval()
                for j, val_ex in enumerate(val_iter):

                    # TODO when debugged: decode everything and measure
                    # * probability of correct transliteration (sum & viterbi)
                    # * edit distance of decoded
                    # * accuracy of decoded

                    with torch.no_grad():
                        print()
                        src_string = "".join(
                            [ar_text_field.vocab.itos[c] for c in val_ex.ar[0]])
                        tgt_string = "".join(
                            [en_text_field.vocab.itos[c] for c in val_ex.en[0]])
                        decoded_val = neural_model.decode(val_ex.ar)
                        hypothesis = "".join(
                            en_text_field.vocab.itos[c] for c in decoded_val[0])

                        correct_probability = neural_model.viterbi(
                            val_ex.ar, val_ex.en)
                        decoded_probability = neural_model.viterbi(
                            val_ex.ar, decoded_val)

                        print(f"{src_string} -> {hypothesis} ({tgt_string})")
                        print(f"  hyp. prob.: {decoded_probability:.3f}, "
                              f"correct prob.: {correct_probability:.3f}, "
                              f"ratio: {decoded_probability / correct_probability:.3f}")

                    if j == 10:
                        break

                print()
                neural_model.train()


if __name__ == "__main__":
    main()
