#!/usr/bin/env python3

import argparse
import datetime

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
    parser.add_argument("--src-tokenized", default=False, action="store_true",
                        help="If true, source side are space separated tokens.")
    parser.add_argument("--tgt-tokenized", default=False, action="store_true",
                        help="If true, target side are space separated tokens.")
    parser.add_argument("--patience", default=20, type=int,
                        help="Number of validations witout improvement before finishing.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ar_text_field, en_text_field, train_iter, val_iter, test_iter = \
        load_transliteration_data(
            args.data_prefix, args.batch_size, device,
            src_tokenized=args.src_tokenized,
            tgt_tokenized=args.tgt_tokenized)

    neural_model = EditDistNeuralModelProgressive(
        ar_text_field.vocab, en_text_field.vocab, device, directed=True,
        encoder_decoder_attention=not args.no_enc_dec_att).to(device)

    kl_div = nn.KLDivLoss(reduction='none').to(device)
    nll = nn.NLLLoss(reduction='none').to(device)
    xent = nn.CrossEntropyLoss(reduction='none').to(device)
    optimizer = optim.Adam(neural_model.parameters())

    step = 0
    best_wer = 1.0
    best_wer_step = 0
    best_cer = 1.0
    best_cer_step = 0
    stalled = 0

    for _ in range(args.epochs):
        if stalled > args.patience:
            break
        for i, train_ex in enumerate(train_iter):
            if stalled > args.patience:
                break
            step += 1

            (action_scores, expected_counts,
                logprob, next_symbol_score) = neural_model(
                train_ex.ar, train_ex.en)

            en_mask = (train_ex.en != neural_model.en_pad).float()
            ar_mask = (train_ex.ar != neural_model.ar_pad).float()
            table_mask = (ar_mask.unsqueeze(2) * en_mask.unsqueeze(1)).float()

            loss = torch.tensor(0.).to(device)
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
                #sampled_actions = torch.multinomial(expected_counts[:, 1:, 1:].reshape(-1, tgt_dim), 1)
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

            loss.backward()

            if step % args.delay_update == args.delay_update - 1:
                stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                print(f"[{stamp}] step: {step}, train loss = {loss:.3g} "
                      f"(NLL {nll_loss:.3g}, "
                      f"EM: {kl_loss:.3g}, "
                      f"sampled EM: {sampled_em_loss:.3g})")
                optimizer.step()
                optimizer.zero_grad()

            if step % (args.delay_update * 50) == args.delay_update * 50 - 1:
                neural_model.eval()

                sources = []
                ground_truth = []
                hypotheses = []

                for j, val_ex in enumerate(val_iter):
                    with torch.no_grad():
                        decoded_val = neural_model.decode(val_ex.ar)

                        for ar, en, hyp in zip(val_ex.ar, val_ex.en, decoded_val):
                            src_string = decode_ids(ar, ar_text_field, args.src_tokenized)
                            tgt_string = decode_ids(en, en_text_field, args.tgt_tokenized)
                            hypothesis = decode_ids(hyp, en_text_field, args.tgt_tokenized)

                            sources.append(src_string)
                            ground_truth.append(tgt_string)
                            hypotheses.append(hypothesis)

                        if j == 0:
                            for src, hyp, tgt in zip(sources[:10], hypotheses, ground_truth):
                                print()
                                print(f"'{src}' -> '{hyp}' ({tgt})")

                print()

                wer = 1 - sum(
                    float(gt == hyp) for gt, hyp
                    in zip(ground_truth, hypotheses)) / len(ground_truth)
                cer = char_error_rate(hypotheses, ground_truth, args.tgt_tokenized)

                stalled += 1
                if wer < best_wer:
                    best_wer = wer
                    best_wer_step = step
                    stalled = 0
                if cer < best_cer:
                    best_cer = cer
                    best_cer_step = step
                    stalled = 0

                print(f"WER: {wer:.3g}   (best {best_wer:.3g}, step {best_wer_step})")
                print(f"CER: {cer:.3g}   (best {best_cer:.3g}, step {best_cer_step})")
                if stalled > 0:
                    print(f"Stalled {stalled} times.")
                print()
                neural_model.train()

    print("TRAINING FINISHED, evaluating on test data")
    print()
    neural_model.eval()

    for j, test_ex in enumerate(test_iter):
        with torch.no_grad():
            decoded_val = neural_model.decode(test_ex.ar)

            for ar, en, hyp in zip(test_ex.ar, test_ex.en, decoded_val):
                src_string = decode_ids(ar, ar_text_field, args.src_tokenized)
                tgt_string = decode_ids(en, en_text_field, args.tgt_tokenized)
                hypothesis = decode_ids(hyp, en_text_field, args.tgt_tokenized)

                sources.append(src_string)
                ground_truth.append(tgt_string)
                hypotheses.append(hypothesis)

            if j == 0:
                for src, hyp, tgt in zip(sources[:10], hypotheses, ground_truth):
                    print()
                    print(f"'{src}' -> '{hyp}' ({tgt})")

    print()

    wer = 1 - sum(
        float(gt == hyp) for gt, hyp
        in zip(ground_truth, hypotheses)) / len(ground_truth)
    cer = char_error_rate(hypotheses, ground_truth, args.tgt_tokenized)

    print(f"WER: {wer:.3g}")
    print(f"CER: {cer:.3g}")
    print()


if __name__ == "__main__":
    main()
