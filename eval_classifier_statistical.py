#!/usr/bin/env python3

import argparse
from collections import defaultdict
import logging
import os

import torch
from torchtext import data

from experiment import experiment_logging, get_timestamp, save_vocab
from statistical_model import EditDistStatModel
from transliteration_utils import decode_ids, char_error_rate


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def load_vocab(file):
    vocab = []
    for token in file:
        vocab.append(token.strip())
    file.close()
    stoi = defaultdict(int)
    for i, symb in enumerate(vocab):
        stoi[symb] = i
    return vocab, stoi

def str_to_vec(stoi, string, tokenized):
    tok = (
        ["<s>"] + (string.split() if tokenized else list(string)) + ["</s>"])

    return torch.tensor([[stoi[s] for s in tok]])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("model", type=argparse.FileType("rb"))
    parser.add_argument("src_vocab", type=argparse.FileType("r"))
    parser.add_argument("tgt_vocab", type=argparse.FileType("r"))
    parser.add_argument("eval_data", type=argparse.FileType("r"))
    parser.add_argument("test_data", type=argparse.FileType("r"))
    parser.add_argument("--src-tokenized", default=False, action="store_true")
    parser.add_argument("--tgt-tokenized", default=False, action="store_true")
    args = parser.parse_args()

    model = torch.load(args.model)
    logging.info("Model loaded.")
    src_vocab, src_stoi = load_vocab(args.src_vocab)
    tgt_vocab, tgt_stoi = load_vocab(args.tgt_vocab)
    logging.info("Vocabularies loaded.")


    pos_scores = []
    neg_scores = []

    logging.info("Estimate threshold on validation data.")
    for i, line in enumerate(args.eval_data):
        src, tgt, clazz = line.strip().split("\t")
        src_idx = str_to_vec(src_stoi, src, args.src_tokenized)
        tgt_idx = str_to_vec(tgt_stoi, tgt, args.tgt_tokenized)

        score = model.viterbi(src_idx, tgt_idx)
        if clazz == "0":
            neg_scores.append(score)
        if clazz == "1":
            pos_scores.append(score)

    threshold = (
        sum(pos_scores) / len(pos_scores) +
        sum(neg_scores) / len(neg_scores)) / 2
    logging.info("Threshold set to %f.", threshold)

    val_true_positives = sum(1.0 for x in pos_scores if x > threshold)
    val_false_positives = sum(1.0 for x in neg_scores if x > threshold)

    if val_true_positives + val_false_positives > 0:
        val_precision = (
            val_true_positives / (val_true_positives + val_false_positives))
    else:
        val_precision = 0
    val_recall = val_true_positives / len(pos_scores)
    if val_precision + val_recall > 0:
        val_f_score = 2 * val_precision * val_recall / (val_precision + val_recall)
    else:
        val_f_score = 0

    logging.info("Validation precision: %f", val_precision)
    logging.info("Validation recall:    %f", val_recall)
    logging.info("Validation F-Score:   %f", val_f_score)

    print(val_f_score)
    exit()

    true_possitive = 0
    real_positives = 0
    all_positives = 0
    all_count = 0
    for i, line in enumerate(args.test_data):
        src, tgt, clazz = line.strip().split("\t")
        src_idx = str_to_vec(src_stoi, src, args.src_tokenized)
        tgt_idx = str_to_vec(tgt_stoi, tgt, args.tgt_tokenized)

        score = model.viterbi(src_idx, tgt_idx)

        if score > threshold:
            all_positives += 1
            if clazz == "1":
                true_possitive += 1
        if clazz == "1":
            real_positives += 1

    if all_positives > 0:
        precision = true_possitive / all_positives
    else:
        precision = 0
    recall = true_possitive / real_positives
    if recall > 0 and precision > 0:
        f_score = 2 * precision * recall / (precision + recall)
    else:
        f_score = 0

    logging.info("Test precision: %f", precision)
    logging.info("Test recall:    %f", recall)
    logging.info("Test F-Score:   %f", f_score)
    print(f_score)


if __name__ == "__main__":
    main()
