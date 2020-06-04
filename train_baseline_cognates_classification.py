#!/usr/bin/env python3

"""Train Tranformer cogantes classifier.

The data directory is expected to contains files {train,eval,text}.txt with
tab-separated values <str1>\\t<str2>\\t{0,1} where 1 is the True label and 0 is
the False label.
"""

import argparse
import logging
import os

import numpy as np
import torch
from torch import optim
from torch.functional import F
from torchtext import data
from transformers import (
    BertForSequenceClassification, BertConfig)

from experiment import experiment_logging, get_timestamp, save_vocab


def cat_examples(dataset, text_field):
    dataset.fields["text"] = text_field
    for example in dataset.examples:
        example.text = example.src + example.tgt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("data_prefix", type=str)
    parser.add_argument("--hidden-size", default=256, type=int)
    parser.add_argument("--attention-heads", default=8, type=int)
    parser.add_argument("--layers", default=4, type=int)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--delay-update", default=1, type=int,
                        help="Update model every N steps.")
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument(
        "--patience", default=10, type=int,
        help="Number of validations witout improvement before finishing.")
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    args = parser.parse_args()

    experiment_params = (
        args.data_prefix.replace("/", "_") +
        f"_hidden{args.hidden_size}" +
        f"_attheads{args.attention_heads}" +
        f"_layers{args.layers}" +
        f"_batch{args.batch_size}" +
        f"_patence{args.patience}" +
        f"_delay{args.delay_update}")
    experiment_dir = experiment_logging(
        f"cognates_class_{experiment_params}_{get_timestamp()}", args)
    model_path = os.path.join(experiment_dir, "model.pt")

    text_field = data.Field(
        tokenize=lambda s: s.split(),
        init_token="<s>", eos_token="</s>", batch_first=True)
    labels_field = data.Field(sequential=False, use_vocab=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, val_data, test_data = data.TabularDataset.splits(
        path=args.data_prefix, train='train.txt',
        validation='eval.txt', test='test.txt', format='tsv',
        fields=[('src', text_field), ('tgt', text_field),
                ('label', labels_field)])

    cat_examples(train_data, text_field)
    cat_examples(val_data, text_field)
    cat_examples(test_data, text_field)

    text_field.build_vocab(train_data)
    save_vocab(
        text_field.vocab.itos, os.path.join(experiment_dir, "src_vocab"))
    save_vocab(
        text_field.vocab.itos, os.path.join(experiment_dir, "tgt_vocab"))

    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train_data, val_data, test_data), batch_sizes=[args.batch_size] * 3,
        shuffle=True, device=device, sort_key=lambda x: len(x.text))

    config = BertConfig(
        vocab_size=len(text_field.vocab),
        is_decoder=False,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.attention_heads,
        intermediate_size=2 * args.hidden_size,
        hidden_act='relu',
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_labels=2)

    model = BertForSequenceClassification(config).to(device)

    optimizer = optim.Adam(model.parameters())

    step = 0
    best_f_score = 0.0
    stalled = 0
    for _ in range(args.epochs):
        if stalled > args.patience:
            break
        for _, train_batch in enumerate(train_iter):
            if stalled > args.patience:
                break
            step += 1

            loss, _ = model(
                train_batch.text.to(device),
                labels=train_batch.label.to(device))
            loss.backward()

            logging.info("step: %d, train loss = %.3g", step, loss)
            optimizer.step()
            optimizer.zero_grad()

            if step % 50 == 49:
                model.eval()
                with torch.no_grad():
                    false_scores = []
                    true_scores = []
                    for j, val_ex in enumerate(val_iter):
                        score = F.softmax(model(val_ex.text)[0], dim=1)[:, 1]
                        for prob, label in zip(score, val_ex.label):
                            if label == 1:
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
                        stalled = 0
                    else:
                        stalled += 1

                    if stalled > 0:
                        logging.info("Stalled %d times.", stalled)

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
            score = F.softmax(model(test_ex.text)[0], dim=1)[:, 1]
            for prob, label in zip(score, test_ex.label):
                if label == 1:
                    true_scores.append(prob.cpu().numpy())
                else:
                    false_scores.append(prob.cpu().numpy())

        pos_mean = np.mean(true_scores)
        neg_mean = np.mean(false_scores)
        boundary = (pos_mean + neg_mean) / 2

        true_positive = np.sum(true_scores > boundary)
        false_positive = np.sum(false_scores > boundary)
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
