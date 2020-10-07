#!/usr/bin/env python3

"""Run Viterbi algorithm string pairs and get best edit operations."""


import argparse
from collections import defaultdict
import logging
import sys

import torch


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
    parser.add_argument("input", type=argparse.FileType("r"), nargs="?",
                        default=sys.stdin)
    parser.add_argument("--src-tokenized", default=False, action="store_true")
    parser.add_argument("--tgt-tokenized", default=False, action="store_true")
    parser.add_argument(
        "--output-format", default="nice", choices=["nice", "tsv", "alignment"],
        help="Nice for command line, tsv for processing, alignment=subtitutitons "
             "in word alignment format.")
    args = parser.parse_args()

    model = torch.load(args.model)
    logging.info("Model loaded.")
    src_vocab, src_stoi = load_vocab(args.src_vocab)
    tgt_vocab, tgt_stoi = load_vocab(args.tgt_vocab)
    logging.info("Vocabularies loaded.")

    for line in args.input:
        line_split = line.strip().split("\t")
        string_1, string_2 = line_split[0], line_split[1]
        string_1_tok = (
            ["<s>"] +
            (string_1.split() if args.src_tokenized else list(string_1)) +
            ["</s>"])
        string_2_tok = (
            ["<s>"] +
            (string_2.split() if args.tgt_tokenized else list(string_2)) +
            ["</s>"])

        string_1_idx = [src_stoi[s] for s in string_1_tok]
        string_2_idx = [tgt_stoi[s] for s in string_2_tok]

        _, edit_ops = model.viterbi(
            torch.tensor([string_1_idx]).cuda(),
            torch.tensor([string_2_idx]).cuda())

        if args.output_format == "alignment":
            alignment = []
            for operation, _, idx in edit_ops[1:-1]:
                if operation == "subs":
                    src_id, tgt_id = idx
                    alignment.append(f"{src_id}-{tgt_id}")
            print(" ".join(alignment))
            continue

        if args.output_format == "nice":
            print(f"{string_1} ⇨ {string_2}")

        readable_ops = []
        for operation, chars, _ in edit_ops[1:-1]:
            if operation == "delete":
                readable_ops.append(f"-{src_vocab[chars]}")
            if operation == "insert":
                readable_ops.append(f"+{tgt_vocab[chars]}")
            if operation == "subs":
                readable_ops.append(
                    f"{src_vocab[chars[0]]}→{tgt_vocab[chars[1]]}")

        if args.output_format == "nice":
            print(" ".join(readable_ops))
            print()
        if args.output_format == "tsv":
            print(string_1, string_2, " ".join(readable_ops), sep="\t")


if __name__ == "__main__":
    main()
