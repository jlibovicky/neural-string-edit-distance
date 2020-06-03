#!/usr/bin/env python3

"""Genenerate heatmaps of alpha tables for given model and data.

The script generates one pdf file per example which is supposed to be a
tab-separated string pair. The PDFs are saved into path <prefix>_{i}.pdf.
"""


import argparse
from collections import defaultdict
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def load_vocab(fh):
    vocab = []
    for token in fh:
        vocab.append(token.strip())
    fh.close()
    stoi = defaultdict(int)
    for i, symb in enumerate(vocab):
        stoi[symb] = i
    return vocab, stoi


def draw(matrix, src, tgt, fig_file):
    fig, ax = plt.subplots()
    im = ax.matshow(np.exp(matrix), cmap='cividis', vmin=0, vmax=1)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)

    ax.set_xlim(-0.5, len(tgt) - 0.5)
    ax.set_ylim(len(src) - 0.5, -0.5)
    ax.set_xticks(np.arange(len(tgt)))
    ax.set_yticks(np.arange(len(src)))
    ax.set_xticklabels(tgt)
    ax.set_yticklabels(src)
    plt.savefig(f"{fig_file}.pdf", bbox_inches='tight',
                pad_inches=0)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("model", type=argparse.FileType("rb"))
    parser.add_argument("src_vocab", type=argparse.FileType("r"))
    parser.add_argument("tgt_vocab", type=argparse.FileType("r"))
    parser.add_argument("prefix", type=str,
                        help="Prefix of the path of the generated files.")
    parser.add_argument("input", type=argparse.FileType("r"), nargs="?",
                        default=sys.stdin)
    parser.add_argument("--src-tokenized", default=False, action="store_true")
    parser.add_argument("--tgt-tokenized", default=False, action="store_true")
    args = parser.parse_args()

    model = torch.load(args.model)
    logging.info("Model loaded.")
    src_vocab, src_stoi = load_vocab(args.src_vocab)
    tgt_vocab, tgt_stoi = load_vocab(args.tgt_vocab)
    logging.info("Vocabularies loaded.")

    for i, line in enumerate(args.input):
        logging.info(f"Inference for example {i + 1}")
        string_1, string_2 = line.strip().split("\t")
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

        alpha = model.alpha(
            torch.tensor([string_1_idx]).cuda(),
            torch.tensor([string_2_idx]).cuda())[0]

        logging.info(f"Generating image.")
        draw(
            alpha.cpu().numpy(), string_1_tok, string_2_tok,
            f"{args.prefix}{i}")


if __name__ == "__main__":
    main()
