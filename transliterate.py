#!/usr/bin/env python3

"""Run trained transliteration model."""


import argparse
from collections import defaultdict
import logging
import sys

import torch

from transliteration_utils import load_vocab, decode_ids, char_error_rate

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("model", type=argparse.FileType("rb"))
    parser.add_argument("src_vocab", type=argparse.FileType("r"))
    parser.add_argument("tgt_vocab", type=argparse.FileType("r"))
    parser.add_argument("input", type=argparse.FileType("r"), nargs="?",
                        default=sys.stdin)
    parser.add_argument("--src-tokenized", default=False, action="store_true")
    parser.add_argument("--tgt-tokenized", default=False, action="store_true")
    parser.add_argument(
        "--decoding", default="greedy",
        choices=["greedy", "beam_search", "operations", "operations_beam"])
    parser.add_argument("--evaluate", default=False, action="store_true")
    parser.add_argument("--beam-size", type=int, default=10)
    args = parser.parse_args()

    model = torch.load(args.model)
    logging.info("Model loaded.")
    src_vocab, src_stoi = load_vocab(args.src_vocab)
    tgt_vocab, tgt_stoi = load_vocab(args.tgt_vocab)
    logging.info("Vocabularies loaded.")

    outputs = []
    targets = []

    for i, line in enumerate(args.input):
        if args.evaluate:
            line_split = line.strip().split("\t")
            string_1, string_2 = line_split[0], line_split[1]
            targets.append(string_2)
        else:
            string_1 = line.strip()
            string_2 = None

        string_1_tok = (
            ["<s>"] +
            (string_1.split() if args.src_tokenized else list(string_1)) +
            ["</s>"])

        string_1_idx = torch.tensor(
            [[src_stoi[s] for s in string_1_tok]]).cuda()

        if args.decoding == "greedy":
            output = model.decode(string_1_idx)
        elif args.decoding == "beam_search":
            output = model.beam_search(string_1_idx, args.beam_size)
        elif args.decoding == "operations":
            output = model.operation_decoding(string_1_idx) #, args.beam_size)
        elif args.decoding == "operations_beam":
            output = model.operation_beam_search(string_1_idx, args.beam_size)
        else:
            raise ValueError(f"Unknown type of decoding: {args.decoding}")

        output_str = decode_ids(output[0], tgt_vocab, args.tgt_tokenized)
        if args.evaluate:
            outputs.append(output_str)
        print(output_str)

        if i % 100 == 99:
            logging.info("Processed %d strings.", i + 1)

    acc = sum(
        float(o == t) for o, t in zip(outputs, targets)) / len(outputs)
    cer = char_error_rate(outputs, targets, args.tgt_tokenized)
    logging.info("WER: %.3g", 1 - acc)
    logging.info("CER: %.3g", cer)


if __name__ == "__main__":
    main()
