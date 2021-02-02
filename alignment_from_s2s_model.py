#!/usr/bin/env python3

"""Get alignment from attention in S2S model."""


import argparse
import logging
import sys
from typing import Dict, List

import torch

from transliteration_utils import load_vocab

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def preprocess_input(
        string: str, tokenized: bool, stoi: Dict[str, int]) -> List[int]:
    string_tok = (
        ["<s>"] + (string.split() if tokenized else list(string)) + ["</s>"])
    string_idx = [stoi[s] for s in string_tok]
    return string_idx


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
        "--threshold", default=0.15, type=float,
        help="Threshold for considering attention as an alignment link.")
    args = parser.parse_args()

    model = torch.load(args.model)
    logging.info("Model loaded.")
    _, src_stoi = load_vocab(args.src_vocab)
    _, tgt_stoi = load_vocab(args.tgt_vocab)
    logging.info("Vocabularies loaded.")

    for i, line in enumerate(args.input):
        string_1, string_2 = line.strip().split("\t")

        string_1_idx = preprocess_input(string_1, args.src_tokenized, src_stoi)
        string_2_idx = preprocess_input(string_2, args.tgt_tokenized, tgt_stoi)

        with torch.no_grad():
            _, attentions = model(
                # pylint: disable=not-callable
                torch.tensor([string_1_idx]).cuda(),
                torch.tensor([string_2_idx]).cuda())
                # pylint: enable=not-callable

        links = []
        soft_alignment = attentions[0, :-1, 1:-1].t()
        for src_idx, row in enumerate(soft_alignment):
            for tgt_idx, val in enumerate(row):
                if val > args.threshold:
                    links.append(f"{src_idx + 1}-{tgt_idx + 1}")
        print(" ".join(links))

        if i % 1000 == 999:
            logging.info("Processed %d pairs.", i + 1)


if __name__ == "__main__":
    main()
