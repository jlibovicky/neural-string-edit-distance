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
    parser.add_argument("--epochs", type=int, default=100)
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
        "--learning-rate", default=0.01, type=float,
        help="Learning rate.")
    parser.add_argument("--log-directory", default="experiments", type=str,
                        help="Number of steps between validations.")
    args = parser.parse_args()

    experiment_params = (
        args.data_prefix.replace("/", "_") +
        "src_tokenized" if args.src_tokenized else "" +
        "tgt_tokenized" if args.tgt_tokenized else "" +
        f"_learning_rate_{args.learning_rate}" +
        f"_patience{args.patience}")
    experiment_dir = experiment_logging(
        args.log_directory,
        f"edit_stat_{experiment_params}_{get_timestamp()}", args)
    model_path = os.path.join(experiment_dir, "model.pt")

    src_text_field = data.Field(
        tokenize=(lambda s: s.split()) if args.src_tokenized else list,
        init_token="<s>", eos_token="</s>", batch_first=True)
    tgt_text_field = data.Field(
        tokenize=(lambda s: s.split()) if args.tgt_tokenized else list,
        init_token="<s>", eos_token="</s>", batch_first=True)

    train_data, val_data, test_data = data.TabularDataset.splits(
        path=args.data_prefix, train='train.txt',
        validation='eval.txt', test='test.txt', format='tsv',
        fields=[('ar', src_text_field), ('en', tgt_text_field)])

    # Use val data beacuse iterating through train data would take agas.
    src_text_field.build_vocab(val_data)
    tgt_text_field.build_vocab(val_data)

    save_vocab(
        src_text_field.vocab.itos, os.path.join(experiment_dir, "src_vocab"))
    save_vocab(
        tgt_text_field.vocab.itos, os.path.join(experiment_dir, "tgt_vocab"))
    logging.info("Loaded data, created vocabularies..")

    # pylint: disable=W0632
    train_iter, val_iter, _ = data.Iterator.splits(
        (train_data, val_data, test_data), batch_sizes=(1, 1, 1),
        shuffle=True, device=0, sort_key=lambda x: len(x.ar))
    # pylint: enable=W0632

    model = EditDistStatModel(src_text_field.vocab, tgt_text_field.vocab)

    smallest_tgttropy = 1e9
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

                if entropy < smallest_tgttropy:
                    smallest_tgttropy = entropy
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
                                val_ex.ar[0], src_text_field)
                            tgt_string = decode_ids(
                                val_ex.en[0], tgt_text_field)

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
