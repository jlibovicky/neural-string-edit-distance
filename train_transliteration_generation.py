#!/usr/bin/env python3

import argparse

import logging
from jiwer import wer
import torch
from torch import nn, optim
from torch.functional import F
from torchtext import data
import transformers

from models import (
    EditDistNeuralModelConcurrent, EditDistNeuralModelProgressive)
from transliteration_utils import (
    load_transliteration_data, decode_ids, char_error_rate)


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("data_prefix", type=str)
    parser.add_argument("--em-loss", default=None, type=float)
    parser.add_argument("--nll-loss", default=None, type=float)
    parser.add_argument("--hidden-size", default=64, type=int)
    parser.add_argument("--attention-heads", default=4, type=int)
    parser.add_argument("--layers", default=2, type=int)
    args = parser.parse_args()

    ar_text_field, en_text_field, train_iter, val_iter, test_iter = \
        load_transliteration_data(args.data_prefix, 32)
    tgt_pad_id = en_text_field.vocab[en_text_field.pad_token]

    neural_model = EditDistNeuralModelProgressive(
        ar_text_field.vocab, en_text_field.vocab, directed=True)
    #neural_model = EditDistNeuralModelConcurrent(
    #    ar_text_field.vocab, en_text_field.vocab, directed=True)

    kl_div = nn.KLDivLoss(reduction='none')
    nll = nn.NLLLoss(reduction='none')
    optimizer = optim.Adam(neural_model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.9)

    en_examples = []
    ar_examples = []
    pos_examples = 0
    for _ in range(10):
        for i, train_ex in enumerate(train_iter):
            pos_examples += 1

            (action_scores, expected_counts,
                logprob, next_symbol_score, log_table_mask) = neural_model(
                train_ex.ar, train_ex.en)

            mask = log_table_mask.exp()
            loss = torch.tensor(0.)
            kl_loss = 0
            if args.em_loss is not None:
                kl_loss = (kl_div(action_scores, expected_counts) * mask).sum() / mask.sum()
                loss += args.em_loss * kl_loss

            nll_loss = 0
            if args.nll_loss is not None:
                tgt_mask = (train_ex.en[:, 1:] != tgt_pad_id).reshape(-1)
                nll_loss_raw = nll(
                    next_symbol_score.reshape(-1, next_symbol_score.size(2)),
                    train_ex.en[:, 1:].reshape(-1))
                masked_loss = torch.where(tgt_mask, nll_loss_raw, torch.zeros_like(nll_loss_raw))
                nll_loss = masked_loss.sum() / tgt_mask.float().sum()
                loss += args.nll_loss * nll_loss

            loss.backward()

            if pos_examples % 1 == 0:
                logging.info(f"train loss = {loss:.3g} "
                             f"(NLL {nll_loss:.3g}, "
                             f"EM: {kl_loss:.3g})")
                optimizer.step()
                optimizer.zero_grad()

            if pos_examples % 2000 == 1999:
                neural_model.eval()

                ground_truth = []
                hypotheses = []

                for j, val_ex in enumerate(val_iter):

                    # TODO when debugged: decode everything and measure
                    # * probability of correct transliteration (sum & viterbi)
                    # * edit distance of decoded
                    # * accuracy of decoded

                    with torch.no_grad():
                        src_string = decode_ids(val_ex.ar[0], ar_text_field)
                        tgt_string = decode_ids(val_ex.en[0], en_text_field)
                        decoded_val = neural_model.decode(val_ex.ar)
                        hypothesis = decode_ids(decoded_val[0], en_text_field)

                        ground_truth.append(tgt_string)
                        hypotheses.append(hypothesis)

                        if j < 5:
                            # correct_prob = neural_model.viterbi(
                            #     val_ex.ar, val_ex.en)
                            # decoded_prob = neural_model.viterbi(
                            #     val_ex.ar, decoded_val)

                            print()
                            print(f"'{src_string}' -> '{hypothesis}' ({tgt_string})")
                            # print(f"  hyp. prob.: {decoded_prob:.3f}, "
                            #       f"correct prob.: {correct_prob:.3f}, "
                            #       f"ratio: {decoded_prob / correct_prob:.3f}")

                        if j >= 50:
                            break

                print()
                accuracy = sum(
                    float(gt == hyp) for gt, hyp
                    in zip(ground_truth, hypotheses)) / len(ground_truth)
                print(f"WER: {1 - accuracy}")
                cer = char_error_rate(hypotheses, ground_truth)
                print(f"CER: {cer}")
                print()
                neural_model.train()
                scheduler.step(cer)


if __name__ == "__main__":
    main()
