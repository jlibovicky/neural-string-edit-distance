#!/usr/bin/env python3

"""Binary classification model using neural string edit distance.

The data directory is expected to contains files {train,eval,text}.txt with
tab-separated values <str1>\\t<str2>\\t{0,1} where 1 is the True label and 0 is
the False label.
"""

import argparse
import logging
import os

import numpy as np
import torch
from torch import nn, optim
from torchtext import data

from experiment import experiment_logging, get_timestamp, save_vocab
from models import EditDistNeuralModelConcurrent


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("data_prefix", type=str)
    parser.add_argument("--share-encoders", type=bool, default=True)
    parser.add_argument("--model-type", default='transformer',
                        choices=["transformer", "rnn", "embeddings", "cnn"])
    parser.add_argument("--embedding-dim", default=64, type=int)
    parser.add_argument("--hidden-size", default=256, type=int)
    parser.add_argument("--attention-heads", default=4, type=int)
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--delay-update", default=1, type=int,
                        help="Update model every N steps.")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--interpretation-loss", default=None, type=float)
    parser.add_argument("--src-tokenized", default=False, action="store_true",
                        help="If true, source side is space-separated.")
    parser.add_argument("--tgt-tokenized", default=False, action="store_true",
                        help="If true, target side are space-separated.")
    parser.add_argument("--patience", default=20, type=int,
                        help="Number of no-improvement validations before "
                             "early finishing.")
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--validation-frequency", default=50, type=int,
                        help="Number of steps between validations.")
    parser.add_argument("--log-directory", default="experiments", type=str,
                        help="Number of steps between validations.")
    args = parser.parse_args()

    # TODO PARAMTERICZE LOSSES
    experiment_params = (
        args.data_prefix.replace("/", "_") +
        f"_model{args.model_type}" +
        f"_hidden{args.hidden_size}" +
        f"_attheads{args.attention_heads}" +
        f"_layers{args.layers}" +
        f"_interpretationLoss{args.interpretation_loss}" +
        f"_batch{args.batch_size}" +
        f"_patence{args.patience}")
    experiment_dir = experiment_logging(
        args.log_directory,
        f"edit_class_{experiment_params}_{get_timestamp()}", args)
    model_path = os.path.join(experiment_dir, "model.pt")
    # tb_writer = SummaryWriter(experiment_dir)

    ar_text_field = data.Field(
        tokenize=(lambda s: s.split()) if args.src_tokenized else list,
        init_token="<s>", eos_token="</s>", batch_first=True)
    if args.share_encoders:
        if args.src_tokenized != args.tgt_tokenized:
            raise ValueError(
                "Source and target must be tokenized the same way when "
                "sharing encoders.")
        en_text_field = ar_text_field
    else:
        en_text_field = data.Field(
            tokenize=(lambda s: s.split()) if args.tgt_tokenized else list,
            init_token="<s>", eos_token="</s>", batch_first=True)
    labels_field = data.Field(sequential=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, val_data, test_data = data.TabularDataset.splits(
        path=args.data_prefix, train='train.txt',
        validation='eval.txt', test='test.txt', format='tsv',
        fields=[('ar', ar_text_field), ('en', en_text_field),
                ('labels', labels_field)])

    ar_text_field.build_vocab(train_data)
    if not args.share_encoders:
        en_text_field.build_vocab(train_data)
    labels_field.build_vocab(train_data)
    true_class_label = labels_field.vocab.stoi['1']

    save_vocab(
        ar_text_field.vocab.itos, os.path.join(experiment_dir, "src_vocab"))
    save_vocab(
        en_text_field.vocab.itos, os.path.join(experiment_dir, "tgt_vocab"))

    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train_data, val_data, test_data), batch_sizes=[args.batch_size] * 3,
        shuffle=True, device=device, sort_key=lambda x: len(x.ar))

    model = EditDistNeuralModelConcurrent(
        ar_text_field.vocab, en_text_field.vocab, device,
        model_type=args.model_type,
        hidden_dim=args.hidden_size,
        hidden_layers=args.layers,
        attention_heads=args.attention_heads,
        share_encoders=args.share_encoders).to(device)

    class_loss = nn.BCELoss()
    kl_div_loss = nn.KLDivLoss(reduction='none')
    xent_loss = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters())

    step = 0
    stalled = 0
    best_f_score = 0
    best_boundary = 0.5
    for _ in range(args.epochs):
        if stalled > args.patience:
            break
        for train_batch in train_iter:
            if stalled > args.patience:
                break
            step += 1

            target = (train_batch.labels == true_class_label).float()
            pos_mask = target.unsqueeze(1).unsqueeze(2)
            neg_mask = 1 - pos_mask

            ar_mask = (train_batch.ar != model.ar_pad).float()
            en_mask = (train_batch.en != model.en_pad).float()

            action_mask = ar_mask.unsqueeze(2) * en_mask.unsqueeze(1)

            action_scores, expected_counts, logprobs, distorted_probs = model(
                train_batch.ar, train_batch.en)

            bce_loss = class_loss(logprobs.exp(), target)
            pos_samples_loss = kl_div_loss(
                action_scores.reshape(-1, 4),
                expected_counts.reshape(-1, 4)).sum(1)
            neg_samples_loss = xent_loss(
                action_scores.reshape(-1, 4),
                torch.full(action_scores.shape[:-1],
                           3, dtype=torch.long).to(device).reshape(-1))

            pos_loss = (
                (action_mask * pos_mask).reshape(-1) * pos_samples_loss).mean()
            neg_loss = (
                (action_mask * neg_mask).reshape(-1) * neg_samples_loss).mean()
            loss = pos_loss + neg_loss + bce_loss

            distortion_loss = 0
            if args.interpretation_loss is not None:
                distortion_loss = (
                    (action_mask * distorted_probs).sum() / action_mask.sum())
                loss += args.interpretation_loss * distortion_loss

            loss.backward()

            logging.info(
                "step: %d, train loss = %.3g (positive: %.3g, negative: %.3g, "
                "BCE: %.3g, " "distortion: %.3g)",
                step, loss, pos_loss, neg_loss, bce_loss, distortion_loss)

            optimizer.step()
            optimizer.zero_grad()

            if step % args.validation_frequency == args.validation_frequency - 1:
                model.eval()
                with torch.no_grad():
                    false_scores = []
                    true_scores = []
                    for j, val_ex in enumerate(val_iter):
                        score = model.probabilities(val_ex.ar, val_ex.en)[0]
                        for prob, label in zip(score, val_ex.labels):
                            if label == true_class_label:
                                true_scores.append(prob.cpu().numpy())
                            else:
                                false_scores.append(prob.cpu().numpy())

                        if j > 10:
                            break

                    pos_mean = np.mean(true_scores)
                    neg_mean = np.mean(false_scores)
                    boundary = (pos_mean + neg_mean) / 2

                    true_positive = np.sum(true_scores > boundary)
                    false_positive = np.sum(false_scores > boundary)
                    precision = (
                        true_positive / (true_positive + false_positive))
                    recall = true_positive / len(true_scores)
                    f_score = 2 * precision * recall / (precision + recall)

                    logging.info("")
                    logging.info("neural true  scores: %.3f +/- %.3f",
                                 pos_mean, np.std(true_scores))
                    logging.info("neural false scores: %.3f +/- %.3f",
                                 neg_mean, np.std(false_scores))
                    logging.info("Precision: %.3f", precision)
                    logging.info("Recall: %.3f", recall)
                    logging.info("F1-score: %.3f", f_score)

                    if f_score > best_f_score:
                        torch.save(model, model_path)
                        best_f_score = f_score
                        best_boundary = boundary
                        stalled = 0
                    else:
                        stalled += 1

                    if stalled > 0:
                        logging.info("Stalled %d times (best F score %.3f)",
                                     stalled, best_f_score)

                    logging.info("")
                model.train()

    logging.info("")
    logging.info("TRANING FINISHED, TESTING")
    logging.info("")

    model = torch.load(model_path)
    model.eval()
    with torch.no_grad():
        false_scores = []
        true_scores = []
        for j, test_ex in enumerate(test_iter):
            score = model.probabilities(test_ex.ar, test_ex.en)[0]
            for prob, label in zip(score, test_ex.labels):
                if label == true_class_label:
                    true_scores.append(prob.cpu().numpy())
                else:
                    false_scores.append(prob.cpu().numpy())

        pos_mean = np.mean(true_scores)
        neg_mean = np.mean(false_scores)

        true_positive = np.sum(true_scores > best_boundary)
        false_positive = np.sum(false_scores > best_boundary)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / len(true_scores)
        f_score = 2 * precision * recall / (precision + recall)

        logging.info("")
        logging.info("neural true  scores: %.3f +/- %.3f",
                     pos_mean, np.std(true_scores))
        logging.info("neural false scores: %.3f +/- %.3f",
                     neg_mean, np.std(false_scores))
        logging.info("Precision: %.3f", precision)
        logging.info("Recall: %.3f", recall)
        logging.info("F1-score: %.3f", f_score)


if __name__ == "__main__":
    main()
