#!/usr/bin/env python3

"""Formulate cognate detection as binary classification."""

import argparse
import csv
from collections import defaultdict
import random
import sys


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("dataset_file", type=argparse.FileType("r"))
    parser.add_argument("--seed", type=int, default=85604)
    args = parser.parse_args()

    random.seed(args.seed)

    per_class_listing = defaultdict(list)
    reader = csv.DictReader(args.dataset_file, dialect='excel-tab')
    for item in reader:
        # per_class_listing[item["COGNATE_CLASS"]].append(
        #     f"{item['DOCULECT']} {item['TOKENS']}")
        per_class_listing[item["COGNATE_CLASS"]].append(item['TOKENS'])

    args.dataset_file.close()
    print("Dataset loaded in memory.", file=sys.stderr)

    classification_examples = []
    for clazz, instances in per_class_listing.items():
        for inst1 in instances:
            for inst2 in instances:
                classification_examples.append(f"{inst1}\t{inst2}\t1")

    n_positive_examples = len(classification_examples)
    print(f"Generated {n_positive_examples} positive examples.",
          file=sys.stderr)

    classes = list(per_class_listing.keys())
    for _ in range(n_positive_examples):
        cls_1 = random.choice(classes)
        cls_2 = cls_1
        while cls_1 == cls_2:
            cls_2 = random.choice(classes)

        inst1 = random.choice(per_class_listing[cls_1])
        inst2 = random.choice(per_class_listing[cls_2])
        classification_examples.append(f"{inst1}\t{inst2}\t0")
    print(f"Generated negative samples.", file=sys.stderr)

    random.shuffle(classification_examples)

    for ex in classification_examples:
        print(ex)


if __name__ == "__main__":
    main()
