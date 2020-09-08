#!/usr/bin/env python
# coding: utf-8

"""Evaluate alignment."""

import argparse


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "ground_truth", type=argparse.FileType("r"),
        help="Ground truth alignment.")
    parser.add_argument(
        "prediction", type=argparse.FileType("r"),
        help="Predicted alignment.")
    args = parser.parse_args()

    precisions = []
    recalls = []
    f_scores = []

    for gt_line, pred_line in zip(args.ground_truth, args.prediction):
        gt_set = set(gt_line.strip().split(" "))
        prediction_set = set(pred_line.strip().split(" "))

        in_both_size = len(gt_set.intersection(prediction_set))

        if in_both_size == 0:
            precisions.append(0.)
            recalls.append(0.)
            f_scores.append(0.)

        recall = in_both_size / len(gt_set)
        recalls.append(recall)
        if prediction_set:
            precision = in_both_size / len(prediction_set)
        else:
            precision = 1.
        precisions.append(precision)

        if recall + precision > 0:
            f_scores.append(2 * recall * precision / (recall + precision))
        else:
            f_scores.append(0.)

    assert len(precisions) == len(recalls)
    assert len(precisions) == len(f_scores)

    mean_precision = 100 * sum(precisions) / len(precisions)
    mean_recall = 100 * sum(recalls) / len(recalls)
    mean_f_score = 100 * sum(f_scores) / len(f_scores)

    print(f"Precision: {mean_precision:.2f}")
    print(f"Recall:    {mean_recall:.2f}")
    print(f"F-Score:   {mean_f_score:.2f}")


if __name__ == "__main__":
    main()
