#!/usr/bin/env python3

import argparse
import logging
import os

import torch
from torchtext import data

from experiment import experiment_logging, get_timestamp, save_vocab
from statistical_model import EditDistStatModel
from transliteration_utils import decode_ids


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("data_prefix", type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--src-tokenized", default=False, action="store_true",
        help="If true, source side are space separated tokens.")
    parser.add_argument(
        "--tgt-tokenized", default=False, action="store_true",
        help="If true, target side are space separated tokens.")
    parser.add_argument(
        "--patience", default=10, type=int,
        help="Early stopping patience.")
    parser.add_argument(
        "--learning-rate", default=0.1, type=float,
        help="Learning rate.")
    args = parser.parse_args()

    experiment_params = (
        args.data_prefix.replace("/", "_") +
        "src_tokenized" if args.src_tokenized else "" +
        "tgt_tokenized" if args.tgt_tokenized else "" +
        f"_learning_rate_{args.learning_rate}" +
        f"_patience{args.patience}")
    experiment_dir = experiment_logging(
        f"edit_stat_{experiment_params}_{get_timestamp()}", args)
    model_path = os.path.join(experiment_dir, "model.pt")

    ar_text_field = data.Field(
        tokenize=(lambda s: s.split()) if args.src_tokenized else list,
        init_token="<s>", eos_token="</s>", batch_first=True)
    en_text_field = data.Field(
        tokenize=(lambda s: s.split()) if args.tgt_tokenized else list,
        init_token="<s>", eos_token="</s>", batch_first=True)

    train_data, val_data, test_data = data.TabularDataset.splits(
        path=args.data_prefix, train='train.txt',
        validation='eval.txt', test='test.txt', format='tsv',
        fields=[('ar', ar_text_field), ('en', en_text_field)])

    ar_text_field.build_vocab(train_data)
    en_text_field.build_vocab(train_data)

    save_vocab(
        ar_text_field.vocab.itos, os.path.join(experiment_dir, "src_vocab"))
    save_vocab(
        en_text_field.vocab.itos, os.path.join(experiment_dir, "tgt_vocab"))
    logging.info("Loaded data, created vocabularies..")

    # pylint: disable=W0632
    train_iter, val_iter, _ = data.Iterator.splits(
        (train_data, val_data, test_data), batch_sizes=(1, 1, 1),
        shuffle=True, device=0, sort_key=lambda x: len(x.ar))
    # pylint: enable=W0632

    model = EditDistStatModel(ar_text_field.vocab, en_text_field.vocab)

    smallest_entropy = 1e9
    examples = 0
    stat_expecations = []
    logging.info("Training starts.")
    best_val_score = 0
    stalled = 0
    for _ in range(args.epochs):
        for train_ex in train_iter:
            examples += 1
            exp_counts = model(train_ex.ar, train_ex.en)
            stat_expecations.append(exp_counts)

            if examples % 10 == 9:
                model.maximize_expectation(
                    stat_expecations, learning_rate=args.learning_rate)
                entropy = -(
                    model.weights * model.weights.exp()).sum()
                logging.info("stat. model entropy = %.10g", entropy.cpu())
                stat_expecations = []

                if entropy < smallest_entropy:
                    smallest_entropy = entropy
                    torch.save(model, model_path)

            if examples % 200 == 199:
                logging.info("")
                logging.info("Validation:")
                with torch.no_grad():
                    total_score = 0
                    val_examples = 0
                    for j, val_ex in enumerate(val_iter):
                        val_examples += 1
                        stat_score = model.viterbi(val_ex.ar, val_ex.en)
                        total_score += stat_score

                        if j < 10:
                            src_string = decode_ids(
                                val_ex.ar[0], ar_text_field)
                            tgt_string = decode_ids(
                                val_ex.en[0], en_text_field)

                            logging.info(
                                "%s -> %s  %f", src_string, tgt_string,
                                stat_score)

                        if j == 10:
                            logging.info("")

                    val_score = total_score / val_examples
                    logging.info(
                        "Validation score: %f", val_score)

                    if val_score > best_val_score:
                        best_val_score = val_score
                        stalled = 0
                        logging.info("New maximum!")
                        torch.save(model, model_path)
                    else:
                        stalled += 1
                        logging.info(
                            "Previous best %f, stalled %d times.",
                            best_val_score, stalled)
                    logging.info("")

                    if stalled > args.patience:
                        break
        if stalled > args.patience:
            break

    logging.info("Training finished, best model was saved.")


if __name__ == "__main__":
    main()
