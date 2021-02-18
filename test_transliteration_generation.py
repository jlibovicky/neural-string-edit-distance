#!/usr/bin/env python3

"""Decode test data and evaluate."""


import argparse
from collections import defaultdict
import logging
import sys

import torch

from train_transliteration_s2s import Seq2SeqModel
from cnn import CNNEncoder
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
    parser.add_argument("input", type=argparse.FileType("r"), nargs="?",
                        default=sys.stdin)
    parser.add_argument("--src-tokenized", default=False, action="store_true")
    parser.add_argument("--tgt-tokenized", default=False, action="store_true")
    parser.add_argument("--beam-size", type=int, default=5,
                        help="Beam size for test data decoding.")
    parser.add_argument("--len-norm", type=float, default=1.0,
                        help="Length normalization factor.")
    parser.add_argument("--output", type=argparse.FileType("w"), default=None)
    args = parser.parse_args()

    model = torch.load(args.model)

    if hasattr(model, 'ar_pad'):
        model.src_pad = model.ar_pad
        model.tgt_pad = model.en_pad
        model.tgt_bos = model.en_bos
        model.tgt_eos = model.en_eos
        model.src_encoder = model.ar_encoder
        model.tgt_encoder = model.en_encoder
        model.tgt_symbol_count = model.en_symbol_count

    if (not isinstance(model, Seq2SeqModel) and
            isinstance(model.src_encoder, CNNEncoder)):
        if not hasattr(model.src_encoder, "layers"):
            pass
        # model.encoder.

    logging.info("Model loaded.")
    _, src_stoi = load_vocab(args.src_vocab)
    tgt_vocab, _ = load_vocab(args.tgt_vocab)
    logging.info("Vocabularies loaded.")

    tgt_references = []
    tgt_hypotheses = []
    for line in args.input:
        line_split = line.strip().split("\t")
        string_1, string_2 = line_split[0], line_split[1]
        tgt_references.append(string_2)

        string_1_tok = (
            ["<s>"] +
            (string_1.split() if args.src_tokenized else list(string_1)) +
            ["</s>"])

        string_1_idx = [src_stoi[s] for s in string_1_tok]

        decoded = model.beam_search(
            # pylint: disable=not-callable
            torch.tensor([string_1_idx]).cuda(),
            # pylint: enable=not-callable
            beam_size=args.beam_size,
            len_norm=args.len_norm)

        if isinstance(decoded, tuple):
            decoded = decoded[0]

        tgt_string = decode_ids(decoded[0], tgt_vocab, args.tgt_tokenized)
        if args.output is not None:
            print(tgt_string, file=args.output)
        tgt_hypotheses.append(tgt_string)

    if args.output is not None:
        args.output.close()

    wer = 1 - sum(
        float(gt == hyp) for gt, hyp
        in zip(tgt_references, tgt_hypotheses)) / len(tgt_hypotheses)
    cer = char_error_rate(
        tgt_hypotheses, tgt_references, args.tgt_tokenized)

    logging.info("WER: %.3g", wer)
    logging.info("CER: %.3g", cer)


if __name__ == "__main__":
    main()
