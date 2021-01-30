#!/usr/bin/env python3

"""Sequence generation using neural string edit distance.

The data directory is expected to contains files {train,eval,text}.txt with
tab-separated source and target strings.
"""

import argparse
import logging
import os

from tensorboardX import SummaryWriter
import torch
from torch import nn, optim

from experiment import experiment_logging, get_timestamp, save_vocab
from models import EditDistNeuralModelProgressive
from transliteration_utils import (
    load_transliteration_data, decode_ids, char_error_rate)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("data_prefix", type=str)
    parser.add_argument("--em-loss", default=None, type=float)
    parser.add_argument("--sampled-em-loss", default=None, type=float)
    parser.add_argument("--nll-loss", default=None, type=float)
    parser.add_argument("--distortion-loss", default=None, type=float)
    parser.add_argument("--final-state-loss", default=None, type=float)
    parser.add_argument("--contrastive-loss", default=None, type=float)
    parser.add_argument("--model-type", default='transformer',
                        choices=["transformer", "rnn", "embeddings", "cnn"])
    parser.add_argument("--embedding-dim", default=256, type=int)
    parser.add_argument("--window", default=3, type=int)
    parser.add_argument("--hidden-size", default=256, type=int)
    parser.add_argument("--attention-heads", default=4, type=int)
    parser.add_argument("--no-enc-dec-att", default=False, action="store_true")
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--beam-size", type=int, default=5,
                        help="Beam size for test data decoding.")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--delay-update", default=4, type=int,
                        help="Update model every N steps.")
    parser.add_argument("--epochs", default=10000, type=int)
    parser.add_argument("--src-tokenized", default=False, action="store_true",
                        help="If true, source side is space-separated.")
    parser.add_argument("--tgt-tokenized", default=False, action="store_true",
                        help="If true, target side is space-separated.")
    parser.add_argument("--patience", default=2, type=int,
                        help="Number of validations witout improvement before "
                             "decreasing the learning rate.")
    parser.add_argument("--lr-decrease-count", default=5, type=int,
                        help="Number learning rate decays before "
                             "early stopping.")
    parser.add_argument("--lr-decrease-ratio", default=0.7, type=float,
                        help="Factor by which the learning rate is decayed.")
    parser.add_argument("--learning-rate", default=1e-4, type=float,
                        help="Initial learning rate.")
    parser.add_argument("--validation-frequency", default=50, type=int,
                        help="Number of steps between validations.")
    parser.add_argument("--log-directory", default="experiments", type=str,
                        help="Number of steps between validations.")
    args = parser.parse_args()

    if (args.nll_loss is None and
            args.em_loss is None and args.sampled_em_loss is None):
        parser.error("No loss was specified.")

    experiment_params = (
        args.data_prefix.replace("/", "_") +
        f"_model{args.model_type}" +
        f"_hidden{args.hidden_size}" +
        f"_attheads{args.attention_heads}" +
        f"_layers{args.layers}" +
        f"_encdecatt{not args.no_enc_dec_att}" +
        f"_window{args.window}" +
        f"_batch{args.batch_size}" +
        f"_dealy{args.delay_update}" +
        f"_patence{args.patience}" +
        f"_nll{args.nll_loss}" +
        f"_EMloss{args.em_loss}" +
        f"_sampledEMloss{args.sampled_em_loss}" +
        f"_finalStateLoss{args.final_state_loss}" +
        f"_distortion{args.distortion_loss}")
    experiment_dir = experiment_logging(
        args.log_directory,
        f"edit_gen_{experiment_params}_{get_timestamp()}", args)
    model_path = os.path.join(experiment_dir, "model.pt")
    tb_writer = SummaryWriter(experiment_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_text_field, tgt_text_field, train_iter, val_iter, test_iter = \
        load_transliteration_data(
            args.data_prefix, args.batch_size, device,
            src_tokenized=args.src_tokenized,
            tgt_tokenized=args.tgt_tokenized)

    save_vocab(
        src_text_field.vocab.itos, os.path.join(experiment_dir, "src_vocab"))
    save_vocab(
        tgt_text_field.vocab.itos, os.path.join(experiment_dir, "tgt_vocab"))

    model = EditDistNeuralModelProgressive(
        src_text_field.vocab, tgt_text_field.vocab, device, directed=True,
        model_type=args.model_type,
        hidden_dim=args.hidden_size,
        hidden_layers=args.layers,
        attention_heads=args.attention_heads,
        window=args.window,
        encoder_decoder_attention=not args.no_enc_dec_att).to(device)

    kl_div = nn.KLDivLoss(reduction='none').to(device)
    nll = nn.NLLLoss(reduction='none').to(device)
    xent = nn.CrossEntropyLoss(reduction='none').to(device)
    bce = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate)

    step = 0
    best_wer = 1e9
    best_wer_step = 0
    best_cer = 1e9
    best_cer_step = 0
    stalled = 0
    learning_rate = args.learning_rate
    remaining_decrease = args.lr_decrease_count

    for _ in range(args.epochs):
        if remaining_decrease <= 0:
            break

        for train_ex in train_iter:
            if stalled > args.patience:
                learning_rate *= args.lr_decrease_ratio
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
                remaining_decrease -= 1
                stalled = 0
                logging.info("Decreasing learning rate to %f.", learning_rate)
            if remaining_decrease <= 0:
                break
            step += 1

            (action_scores, expected_counts,
             logprob, next_symbol_score, distorted_probs, contrastive_logprob) = model(
                 train_ex.ar, train_ex.en,
                 contrastive_probs=args.contrastive_loss is not None)

            tgt_mask = (train_ex.en != model.tgt_pad).float()
            src_mask = (train_ex.ar != model.src_pad).float()
            table_mask = (src_mask.unsqueeze(2) * tgt_mask.unsqueeze(1)).float()

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
                # sampled_actions = torch.multinomial(
                #   expected_counts[:, 1:, 1:].reshape(-1, tgt_dim), 1)
                sampled_em_loss_raw = xent(
                    action_scores[:, 1:, 1:].reshape(-1, tgt_dim),
                    sampled_actions[:, 1:, 1:].reshape(-1))
                sampled_em_loss = (
                    (sampled_em_loss_raw *
                     table_mask[:, 1:, 1:].reshape(-1)).sum() /
                    table_mask.sum())
                loss += args.sampled_em_loss * sampled_em_loss

            nll_loss = 0
            if args.nll_loss is not None:
                tgt_dim = next_symbol_score.size(-1)
                nll_loss_raw = nll(
                    next_symbol_score.reshape(-1, tgt_dim),
                    train_ex.en[:, 1:].reshape(-1))
                nll_loss = (
                    (tgt_mask[:, 1:].reshape(-1) * nll_loss_raw).sum() /
                    tgt_mask[:, 1:].sum())
                loss += args.nll_loss * nll_loss

            distortion_loss = 0
            if args.distortion_loss is not None:
                distortion_loss = (
                    (table_mask * distorted_probs).sum() / table_mask.sum())
                loss += args.distortion_loss * distortion_loss

            final_state_loss = 0
            if args.final_state_loss is not None:
                final_state_loss = bce(logprob.exp(), torch.ones_like(logprob))
                loss += args.final_state_loss * final_state_loss

            contrastive_loss = 0
            if args.contrastive_loss is not None:
                contrastive_loss = bce(
                    contrastive_logprob.exp(),
                    torch.zeros_like(contrastive_logprob))
                loss += args.contrastive_loss * contrastive_loss

            loss.backward()

            if step % args.delay_update == args.delay_update - 1:
                logging.info(
                    "step: %d, train loss = %.3g "
                    "(NLL %.3g, distortion: %.3g, "
                    "final state NLL: %.3g, "
                    "final state contr: %.3g, "
                    "EM: %.3g, sampled EM: %.3g)",
                    step, loss, nll_loss, distortion_loss, final_state_loss,
                    contrastive_loss, kl_loss, sampled_em_loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if (step % (args.delay_update * args.validation_frequency) ==
                    args.delay_update * args.validation_frequency - 1):
                tb_writer.add_scalar('train/loss', loss, step)
                tb_writer.add_scalar('train/nll', nll_loss, step)
                tb_writer.add_scalar('train/em_kl_div', kl_loss, step)
                tb_writer.add_scalar(
                    'train/sampled_em_nll', sampled_em_loss, step)
                model.eval()

                sources = []
                ground_truth = []
                hypotheses = []

                for j, val_ex in enumerate(val_iter):
                    with torch.no_grad():
                        decoded_val = model.decode(val_ex.ar)

                        for ar, en, hyp in zip(
                                val_ex.ar, val_ex.en, decoded_val):
                            src_string = decode_ids(
                                ar, src_text_field, args.src_tokenized)
                            tgt_string = decode_ids(
                                en, tgt_text_field, args.tgt_tokenized)
                            hypothesis = decode_ids(
                                hyp, tgt_text_field, args.tgt_tokenized)

                            sources.append(src_string)
                            ground_truth.append(tgt_string)
                            hypotheses.append(hypothesis)

                        if j == 0:
                            for src, hyp, tgt in zip(
                                    sources[:10], hypotheses, ground_truth):
                                logging.info("")
                                logging.info(
                                    "'%s' -> '%s' (%s)", src, hyp, tgt)

                logging.info("")

                wer = 1 - sum(
                    float(gt == hyp) for gt, hyp
                    in zip(ground_truth, hypotheses)) / len(ground_truth)
                cer = char_error_rate(
                    hypotheses, ground_truth, args.tgt_tokenized)

                stalled += 1
                if wer < best_wer:
                    best_wer = wer
                    best_wer_step = step
                    stalled = 0
                if cer < best_cer:
                    best_cer = cer
                    best_cer_step = step
                    stalled = 0

                logging.info(
                    "WER: %.3g   (best %.3g, step %d)",
                    wer, best_wer, best_wer_step)
                logging.info(
                    "CER: %.3g   (best %.3g, step %d)",
                    cer, best_cer, best_cer_step)
                if stalled > 0:
                    logging.info("Stalled %d times.", stalled)
                else:
                    torch.save(model, model_path)

                logging.info("")

                tb_writer.add_scalar('val/cer', cer, step)
                tb_writer.add_scalar('val/wer', wer, step)
                tb_writer.flush()
                model.train()

    logging.info("TRAINING FINISHED, evaluating on test data")
    logging.info("")
    model = torch.load(model_path)
    model.eval()

    sources = []
    ground_truth = []
    hypotheses = []

    for j, test_ex in enumerate(test_iter):
        with torch.no_grad():
            decoded_val = model.beam_search(
                test_ex.ar, beam_size=args.beam_size)

            for ar, en, hyp in zip(test_ex.ar, test_ex.en, decoded_val):
                src_string = decode_ids(ar, src_text_field, args.src_tokenized)
                tgt_string = decode_ids(en, tgt_text_field, args.tgt_tokenized)
                hypothesis = decode_ids(hyp, tgt_text_field, args.tgt_tokenized)

                sources.append(src_string)
                ground_truth.append(tgt_string)
                hypotheses.append(hypothesis)

            if j == 0:
                for src, hyp, tgt in zip(
                        sources[:10], hypotheses, ground_truth):
                    logging.info("")
                    logging.info("'%s' -> '%s' (%s)", src, hyp, tgt)

    logging.info("")

    wer = 1 - sum(
        float(gt == hyp) for gt, hyp
        in zip(ground_truth, hypotheses)) / len(ground_truth)
    cer = char_error_rate(hypotheses, ground_truth, args.tgt_tokenized)

    logging.info("WER: %.3g", wer)
    logging.info("CER: %.3g", cer)
    logging.info("")


if __name__ == "__main__":
    main()
