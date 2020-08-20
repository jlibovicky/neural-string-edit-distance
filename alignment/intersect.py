#!/usr/bin/env python3

"""Intersect of forward and backward alignment."""


import argparse


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("forward", type=argparse.FileType("r"))
    parser.add_argument("backward", type=argparse.FileType("r"))
    args = parser.parse_args()

    for fw_line, bw_line in zip(args.forward, args.backward):
        fw_tokens = fw_line.strip().split(" ")
        bw_tokens = set(bw_line.strip().split(" "))

        intersection = []
        for tok in fw_tokens:
            if tok in bw_tokens:
                intersection.append(tok)

        print(" ".join(intersection))


if __name__ == "__main__":
    main()
