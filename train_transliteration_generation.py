#!/usr/bin/env python3

import argparse

import torch
from torch import nn, optim
from torch.functional import F
from torchtext import data

from models import (
    EditDistNeuralModelConcurrent, EditDistNeuralModelProgressive)
from transliteration_utils import (
    load_transliteration_data, decode_ids, char_error_rate)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("data_prefix", type=str)
    parser.add_argument("--em-loss", default=None, type=float)
    parser.add_argument("--sampled-em-loss", default=None, type=float)
    parser.add_argument("--nll-loss", default=None, type=float)
    parser.add_argument("--s2s-loss", default=None, type=float)
    parser.add_argument("--hidden-size", default=64, type=int)
    parser.add_argument("--attention-heads", default=4, type=int)
    parser.add_argument("--no-enc-dec-att", default=False, action="store_true")
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--delay-update", default=1, type=int,
                        help="Update model every N steps.")
    parser.add_argument("--epochs", default=10, type=int)
    args = parser.parse_args()

    ar_text_field, en_text_field, train_iter, val_iter, test_iter = \
        load_transliteration_data(args.data_prefix, args.batch_size)

    neural_model = EditDistNeuralModelProgressive(
        ar_text_field.vocab, en_text_field.vocab, directed=True,
        encoder_decoder_attention=not args.no_enc_dec_att)

    kl_div = nn.KLDivLoss(reduction='none')
    nll = nn.NLLLoss(reduction='none')
    xent = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(neural_model.parameters())

    step = 0
    best_wer = 1.0
    best_wer_step = 0
    best_cer = 1.0
    best_cer_step = 0

    for _ in range(args.epochs):
        for i, train_ex in enumerate(train_iter):
            step += 1

            (action_scores, expected_counts,
                logprob, next_symbol_score, seq2seq_logits) = neural_model(
                train_ex.ar, train_ex.en)

            en_mask = (train_ex.en != neural_model.en_pad).float()
            ar_mask = (train_ex.ar != neural_model.ar_pad).float()
            table_mask = (ar_mask.unsqueeze(2) * en_mask.unsqueeze(1)).float()

            loss = torch.tensor(0.)
            kl_loss = 0
            if args.em_loss is not None:
                tgt_dim = action_scores.size(-1)
                kl_loss_raw = kl_div(
                    action_scores.reshape(-1, tgt_dim),
                    expected_counts.reshape(-1, tgt_dim)).sum(1)
                kl_loss = (
                    (kl_loss_raw * table_mask.reshape(-1)).sum() /
                    table_mask.sum())
                loss += args.em_loss * kl_loss

            sampled_em_loss = 0
            if args.sampled_em_loss is not None:
                tgt_dim = action_scores.size(-1)
                # TODO do real sampling instead of argmax
                sampled_actions = expected_counts.argmax(3)
                sampled_em_loss_raw = xent(
                    action_scores[:, 1:, 1:].reshape(-1, tgt_dim),
                    sampled_actions[:, 1:, 1:].reshape(-1))
                sampled_em_loss = (
                    (sampled_em_loss_raw * table_mask[:, 1:, 1:].reshape(-1)).sum() /
                    table_mask.sum())
                loss += args.sampled_em_loss * sampled_em_loss

            nll_loss = 0
            if args.nll_loss is not None:
                tgt_dim = next_symbol_score.size(-1)
                nll_loss_raw = nll(
                    next_symbol_score.reshape(-1, tgt_dim),
                    train_ex.en[:, 1:].reshape(-1))
                nll_loss = (
                    (en_mask[:, 1:].reshape(-1) * nll_loss_raw).sum() /
                    en_mask[:, 1:].sum())
                loss += args.nll_loss * nll_loss

            seq2seq_loss = 0
            if args.s2s_loss is not None:
                tgt_dim = seq2seq_logits.size(-1)
                seq2seq_loss_raw = xent(
                    seq2seq_logits[:, :-1].reshape(-1, tgt_dim),
                    train_ex.en[:, 1:].reshape(-1))
                seq2seq_loss = (
                    (en_mask[:, 1:].reshape(-1) * seq2seq_loss_raw).sum() /
                    en_mask[:, 1:].sum())
                loss += args.s2s_loss * seq2seq_loss

            loss.backward()

            if step % args.delay_update == args.delay_update - 1:
                print(f"step: {step}, train loss = {loss:.3g} "
                      f"(NLL {nll_loss:.3g}, "
                      f"EM: {kl_loss:.3g}, "
                      f"sampled EM: {sampled_em_loss:.3g}, "
                      f"S2S: {seq2seq_loss:.3g})")
                optimizer.step()
                optimizer.zero_grad()

            if step % (args.delay_update * 50) == args.delay_update * 50 - 1:
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

                wer = 1 - sum(
                    float(gt == hyp) for gt, hyp
                    in zip(ground_truth, hypotheses)) / len(ground_truth)
                cer = char_error_rate(hypotheses, ground_truth)

                if wer < best_wer:
                    best_wer = wer
                    best_wer_step = step
                if cer < best_cer:
                    best_cer = cer
                    best_cer_step = step

                print(f"WER: {wer:.3g}   (best {best_wer:.3g}, step {best_wer_step})")
                print(f"CER: {cer:.3g}   (best {best_cer:.3g}, step {best_cer_step})")
                print()
                neural_model.train()


if __name__ == "__main__":
    main()
