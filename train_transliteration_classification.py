#!/usr/bin/env python3

import argparse
import logging
import os

import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.functional import F
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
    parser.add_argument("--src-tokenized", default=False, action="store_true",
                        help="If true, source side are space separated tokens.")
    parser.add_argument("--tgt-tokenized", default=False, action="store_true",
                        help="If true, target side are space separated tokens.")
    parser.add_argument("--patience", default=20, type=int,
                        help="Number of validations witout improvement before finishing.")
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    args = parser.parse_args()

    # TODO PARAMTERICZE LOSSES
    experiment_params = (
        args.data_prefix.replace("/", "_") +
        f"_model{args.model_type}" +
        f"_hidden{args.hidden_size}" +
        f"_attheads{args.attention_heads}" +
        f"_layers{args.layers}" +
        f"_batch{args.batch_size}" +
        f"_patence{args.patience}")
    experiment_dir = experiment_logging(
        f"edit_class_{experiment_params}_{get_timestamp()}", args)
    model_path = os.path.join(experiment_dir, "model.pt")
    tb_writer = SummaryWriter(experiment_dir)

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

    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train_data, val_data, test_data), batch_sizes=[args.batch_size] * 3,
        shuffle=True, device=device, sort_key=lambda x: len(x.ar))

    model = EditDistNeuralModelConcurrent(
        ar_text_field.vocab, en_text_field.vocab, device,
        model_type=args.model_type,
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
        for i, train_batch in enumerate(train_iter):
            if stalled > args.patience:
                break
            step += 1

            target = (train_batch.labels == true_class_label).float()
            pos_mask = target.unsqueeze(1).unsqueeze(2)
            neg_mask = 1 - pos_mask

            ar_mask = (train_batch.ar != model.ar_pad).float()
            en_mask = (train_batch.en != model.en_pad).float()

            action_mask = ar_mask.unsqueeze(2) * en_mask.unsqueeze(1)

            action_scores, expected_counts, logprobs = model(
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

            loss.backward()

            logging.info(f"step: {step}, train loss = {loss:.3g} "
                         f"(positive: {pos_loss:.3g}, "
                         f"negative: {neg_loss:.3g}, "
                         f"BCE: {bce_loss:.3g})")
            optimizer.step()
            optimizer.zero_grad()

            if step % 50 == 49:
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
                    precision = true_positive / (true_positive + false_positive)
                    recall = true_positive / len(true_scores)
                    f_score = 2 * precision * recall / (precision + recall)

                    logging.info("")
                    logging.info(f"neural true  scores: {pos_mean:.3f} +/- {np.std(true_scores):.3f}")
                    logging.info(f"neural false scores: {neg_mean:.3f} +/- {np.std(false_scores):.3f}")
                    logging.info(f"Precision: {precision:.3f}")
                    logging.info(f"Recall: {recall:.3f}")
                    logging.info(f"F1-score: {f_score:.3f}")

                    if f_score > best_f_score:
                        torch.save(model, model_path)
                        best_f_score = f_score
                        best_boundary = boundary
                        stalled = 0
                    else:
                        stalled += 1

                    if stalled > 0:
                        logging.info(
                            f"Stalled {stalled} times (best F score {best_f_score})")

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
        logging.info(f"neural true  scores: {pos_mean:.3f} +/- {np.std(true_scores):.3f}")
        logging.info(f"neural false scores: {neg_mean:.3f} +/- {np.std(false_scores):.3f}")
        logging.info(f"Precision: {precision:.3f}")
        logging.info(f"Recall: {recall:.3f}")
        logging.info(f"F1-score: {f_score:.3f}")


if __name__ == "__main__":
    main()
