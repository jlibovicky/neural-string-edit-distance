#!/usr/bin/env python3

"""Evalute model for detection cogantes.

Apply a detection model on all pairs of test words. This gives a complete graph
that is partitioned using the InfoMap algorithm. The graph particions are
treated as clustering and evaluated against the cognate groups annotated in the
data.
"""

import argparse
from collections import defaultdict
import csv
import logging
import multiprocessing

import bcubed
import infomap
import igraph
from progress.bar import Bar
import torch
import torch.nn as nn
from torch.functional import F
from transformers import BertForSequenceClassification

from models import EditDistStatModel


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def load_vocab(fh):
    vocab_itos = [line.strip() for line in fh]
    fh.close()
    vocab_stoi = defaultdict(int)
    for i, sym in enumerate(vocab_itos):
        vocab_stoi[sym] = i
    return vocab_itos, vocab_stoi


def word_to_tensor(word, vocab_stoi):
    return torch.tensor([[vocab_stoi[tok] for tok in f"<s> {word} </s>".split()]])


def get_grouping(all_scores, words, threshold):
    im = infomap.Infomap("--two-level")

    for i, _ in enumerate(words):
        im.add_node(i)

    for (i, j), score in all_scores.items():
        if score > threshold:
            im.add_link(i, j, score)
    im.run()
    learned_grouping = {}
    for node in im.tree:
        if node.is_leaf:
            word = words[node.node_id][0]
            learned_grouping[word] = set([f"cluster_{node.module_id}"])

    return learned_grouping, im


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("model", type=argparse.FileType("rb"))
    parser.add_argument("src_vocab", type=argparse.FileType("r"))
    parser.add_argument("tgt_vocab", type=argparse.FileType("r"))
    parser.add_argument("test_data", type=argparse.FileType("r"))
    parser.add_argument("--batch", type=int, default=30000)
    args = parser.parse_args()

    src_vocab_itos, src_vocab_stoi = load_vocab(args.src_vocab)
    tgt_vocab_itos, tgt_vocab_stoi = load_vocab(args.tgt_vocab)
    logging.info("Loaded vocabulary.")

    torch.set_num_threads(20)
    model = torch.load(args.model)
    logging.info("Loaded model.")

    words = []
    reader = csv.DictReader(args.test_data, dialect='excel-tab')
    for item in reader:
        words.append((item['TOKENS'], set([item["COGNATE_CLASS"]])))
    args.test_data.close()
    words = words[:2000]
    logging.info("Loaded test data, computing scores.")

    bar = Bar(
        'Scoring', max=(len(words) ** 2 - len(words)) / 2 / args.batch,
        suffix = '%(index)d/%(max)d %(percent).1f%% - %(eta)ds')
    batch = []
    all_scores = {}

    def score_batch():
        if isinstance(model, EditDistStatModel):
            def run_batch(batch):
                out_scores = []
                for _, _, w1, w2 in batch:
                    out_scores.append(model.viterbi(w1, w2).numpy().tolist())
                return out_scores

            ps = len(batch) // 2
            batched_data = [batch[i * ps:(i + 1) * ps] for i in range(2)]
            batch_scores = parmap(run_batch, batched_data, 2)
            scores = []
            for b in batch_scores:
                for s in b:
                    scores.append(s)
        elif isinstance(model, BertForSequenceClassification):
            padded = nn.utils.rnn.pad_sequence(
                [torch.cat((x[2][0], x[3][0]), dim=0) for x in batch], batch_first=True)
            scores = F.softmax(model(padded.cuda())[0], dim=1)[:, 1].cpu().numpy()
        else:
            src_padded = nn.utils.rnn.pad_sequence(
                [x[2][0] for x in batch], batch_first=True)
            tgt_padded = nn.utils.rnn.pad_sequence(
                [x[3][0] for x in batch], batch_first=True)
            raise NotImplemented()

        for (ii, jj, _, _), score in zip(batch, scores):
            all_scores[(ii, jj)] = score

    for i, (word_i, _) in enumerate(words):
        for j, (word_j, _) in enumerate(words):
            if i >= j:
                continue
            word_i_tensor = word_to_tensor(word_i, src_vocab_stoi)
            word_j_tensor = word_to_tensor(word_j, tgt_vocab_stoi)
            batch.append((i, j, word_i_tensor, word_j_tensor))

            if len(batch) >= args.batch:
                score_batch()
                batch = []

                bar.next()
    score_batch()
    bar.next()
    bar.finish()

    ground_truh_grouping = dict(words)
    learned_groupings = []
    ims = []

    thresholds = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    for threshold in thresholds:
        logging.info(f"Threshold {threshold}")
        logging.info("Running InfoMap.")
        grouping, im = get_grouping(all_scores, words, threshold)
        learned_groupings.append(grouping)
        ims.append(im)

    logging.info("Evaluating.")
    for thres, learned_grouping in zip(thresholds, learned_groupings):
        precision = bcubed.precision(ground_truh_grouping, learned_grouping)
        recall = bcubed.recall(ground_truh_grouping, learned_grouping)
        fscore = bcubed.fscore(precision, recall)

        logging.info(f"Threshold {thres}")
        logging.info(f"Precision:  {100 * precision:.3f}")
        logging.info(f"Recall:     {100 * recall:.3f}")
        logging.info(f"F-Score:    {100 * fscore:.3f}")
        logging.info("")



# SOME STUFF COPIED FROM STACKOVERFLOW TO BE ABLE TO MAP WITH CLOSURE FUNCTION
# https://stackoverflow.com/questions/3288595/multiprocessing-how-to-use-pool-map-on-a-function-defined-in-a-class
def fun(f, q_in, q_out, total_size):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))
        #print(f"Progress: {q_out.qsize()} / {total_size} "
        #      f"({100 * q_out.qsize() / total_size:.0f}%)",
        #      end="\r", file=sys.stderr)


def parmap(f, X, nprocs, keep_order=True):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out, len(X)))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    if keep_order:
        return [x for i, x in sorted(res)]
    return [x for i, x in res]
# END OF STACKOVERFLOW COPY-PASTE


if __name__ == "__main__":
    main()
