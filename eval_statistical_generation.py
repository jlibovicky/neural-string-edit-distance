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


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("model", type=argparse.FileType("rb"))
    parser.add_argument("src_vocab", type=argparse.FileType("r"))
    parser.add_argument("tgt_vocab", type=argparse.FileType("r"))
    parser.add_argument("data", type=argparse.FileType("r"))
    parser.add_argument("--src-tokenized", default=False, action="store_true")
    parser.add_argument("--tgt-tokenized", default=False, action="store_true")
    args = parser.parse_args()

    model = torch.load(args.model)
    logging.info("Model loaded.")
    src_vocab, src_stoi = load_vocab(args.src_vocab)
    tgt_vocab, tgt_stoi = load_vocab(args.tgt_vocab)
    logging.info("Vocabularies loaded.")

    correct = 0
    total = 0

    references = []
    hypotheses = []

    for i, line in enumerate(args.data):
        total += 1
        src, tgt = line.strip().split("\t")
        src_tok = (
            ["<s>"] +
            (src.split() if args.src_tokenized else list(src)) +
            ["</s>"])
        tgt_tok = (
            ["<s>"] +
            (tgt.split() if args.tgt_tokenized else list(tgt)) +
            ["</s>"])

        src_idx = torch.tensor([[src_stoi[s] for s in src_tok]])
        tgt_idx = torch.tensor([[tgt_stoi[s] for s in tgt_tok]])

        hyp = [tgt_vocab[idx] for idx in model.decode(src_idx)]
        hyp_str = " ".join(hyp) if args.tgt_tokenized else "".join(hyp)
        correct += hyp_str == tgt

        references.append(tgt)
        hypotheses.append(hyp_str)

        if i < 10:
            logging.info("'%s' -> '%s' (%s)", src, hyp_str, tgt)

    wer = 1 - correct / total
    logging.info("WER: %.3f", wer)
    cer = char_error_rate(hypotheses, references, tokenized=args.tgt_tokenized)
    logging.info("CER: %.3f", cer)
    print(cer, wer)

if __name__ == "__main__":
    main()
