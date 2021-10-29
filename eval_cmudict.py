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
    parser.add_argument("test_set", type=argparse.FileType("r"), nargs="?",
                        default=sys.stdin)
    parser.add_argument(
        "--decoding", default="beam_search",
        choices=["greedy", "beam_search", "operations", "operations_beam"])
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--len-norm", type=float, default=1.6)
    args = parser.parse_args()

    model = torch.load(args.model)
    logging.info("Model loaded.")
    src_vocab, src_stoi = load_vocab(args.src_vocab)
    tgt_vocab, tgt_stoi = load_vocab(args.tgt_vocab)
    logging.info("Vocabularies loaded.")

    outputs = []
    targets = []

    logging.info("Collecting test data.")
    test_data = {}
    for line in args.test_set:
        src, tgt = line.strip().split("\t")
        if src not in test_data:
            test_data[src] = [tgt]
        else:
            test_data[src].append(tgt)
    args.test_set.close()

    logging.info("Generate transcriptions.")
    cers = []
    for i, (src, tgts) in enumerate(test_data.items()):
        src_tok = ["<s>"] + list(src) + ["</s>"]

        src_idx = torch.tensor(
            [[src_stoi[s] for s in src_tok]]).cuda()

        if args.decoding == "greedy":
            output = model.decode(src_idx)
        elif args.decoding == "beam_search":
            output = model.beam_search(src_idx, args.beam_size, len_norm=args.len_norm)
        elif args.decoding == "operations":
            output = model.operation_decoding(src_idx) #, args.beam_size)
        elif args.decoding == "operations_beam":
            output = model.operation_beam_search(src_idx, args.beam_size)
        else:
            raise ValueError(f"Unknown type of decoding: {args.decoding}")

        output_str = decode_ids(output[0], tgt_vocab, tokenized=True)

        cer = min(char_error_rate(
            [output_str] * len(tgts), tgts, tokenized=True, average=False))
        cers.append(cer)

        if i % 100 == 99:
            logging.info("Processed %d / %d strings.", i + 1, len(test_data))

    final_acc = sum(float(cer == 0) for cer in cers) / len(cers)
    final_cer = sum(cers) / len(cers)
    logging.info("WER: %.3g", 1 - final_acc)
    logging.info("CER: %.3g", final_cer)
    print(1 - final_acc)
    print(final_cer)


if __name__ == "__main__":
    main()
