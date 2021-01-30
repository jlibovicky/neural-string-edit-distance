#!/usr/bin/env python3

"""Train a STANCE classifier.

The data directory is expected to contains files {train,eval,text}.txt with
tab-separated values <str1>\\t<str2>\\t{0,1} where 1 is the True label and 0 is
the False label.
"""

import argparse
import logging
import os

import torch
from torch import optim
from torchtext import data
from transformers import BertModel, BertConfig

from experiment import experiment_logging, get_timestamp, save_vocab
from cnn import CNNEncoder
from rnn import RNNEncoder
from stance import Stance


def eval_model(encoder, model, data_iter, threshold, device):
    true_positives = 0
    false_positives = 0
    ground_truth_positives = 0
    ground_truth_negatives = 0

    pos_scores_sum = 0
    neg_scores_sum = 0

    for batch in data_iter:
        with torch.no_grad():
            query = batch.src.to(device)
            query_mask = query != 1
            pos = batch.tgt.to(device)
            pos_mask = pos != 1

            encoded_query = encoder(
                query, attention_mask=query_mask)[0]
            encoded_pos = encoder(pos, attention_mask=pos_mask)[0]
            scores = model.score_pair(
                encoded_query, encoded_pos,
                query_mask, pos_mask)

            ground_truth_positives += batch.label.sum()
            ground_truth_negatives += (1 - batch.label).sum()
            pos_scores_sum += (batch.label * scores).sum()
            neg_scores_sum += ((1 - batch.label) * scores).sum()


            model_positives = (scores > threshold).long()
            true_positives += (
                batch.label * model_positives).sum()
            false_positives += (
                (1 - batch.label) * model_positives).sum()

    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = (
            true_positives.float() /
            (true_positives + false_positives))
    recall = true_positives.float() / ground_truth_positives
    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * precision * recall / (precision + recall)

    pos_score_avg = pos_scores_sum / ground_truth_positives
    neg_score_avg = neg_scores_sum / ground_truth_negatives

    return f_score, precision, recall, pos_score_avg, neg_score_avg


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("data_prefix", type=str)
    parser.add_argument("--tokenized", default=False, action="store_true",
                        help="If true, input is space-separated.")
    parser.add_argument("--hidden-size", default=256, type=int)
    parser.add_argument("--attention-heads", default=8, type=int)
    parser.add_argument("--layers", default=4, type=int)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--delay-update", default=1, type=int,
                        help="Update model every N steps.")
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--model-type", default='transformer',
                        choices=["transformer", "rnn", "embeddings", "cnn"])
    parser.add_argument(
        "--patience", default=10, type=int,
        help="Number of validations witout improvement before finishing.")
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--lr-decrease-count", default=10, type=int,
                        help="Number learning rate decays before "
                             "early stopping.")
    parser.add_argument("--lr-decrease-ratio", default=0.7, type=float,
                        help="Factor by which the learning rate is decayed.")
    parser.add_argument("--log-directory", default="experiments", type=str,
                        help="Number of steps between validations.")
    args = parser.parse_args()

    experiment_params = (
        args.data_prefix.replace("/", "_") +
        f"_model{args.model_type}" +
        f"_hidden{args.hidden_size}" +
        f"_attheads{args.attention_heads}" +
        f"_layers{args.layers}" +
        f"_batch{args.batch_size}" +
        f"_patence{args.patience}" +
        f"_delay{args.delay_update}")
    experiment_dir = experiment_logging(
        args.log_directory,
        f"stance_{experiment_params}_{get_timestamp()}", args)
    model_path = os.path.join(experiment_dir, "model.pt")

    text_field = data.Field(
        tokenize=(lambda s: s.split()) if args.tokenized else list,
        init_token="<s>", eos_token="</s>", batch_first=True)
    labels_field = data.Field(sequential=False, use_vocab=False)
    labels_field = data.Field(sequential=False, use_vocab=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = data.TabularDataset.splits(
        path=args.data_prefix, train='train_stance.txt',
        format='tsv',
        fields=[('src', text_field), ('tgt', text_field), ('neg', text_field)])[0]
    val_data, test_data = data.TabularDataset.splits(
        path=args.data_prefix,
        validation='eval.txt', test='test.txt', format='tsv',
        fields=[('src', text_field), ('tgt', text_field), ('label', labels_field)])

    # Use val data beacuse iterating through train data would take agas.
    text_field.build_vocab(val_data)
    save_vocab(
        text_field.vocab.itos, os.path.join(experiment_dir, "src_vocab"))
    save_vocab(
        text_field.vocab.itos, os.path.join(experiment_dir, "tgt_vocab"))

    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train_data, val_data, test_data), batch_sizes=[args.batch_size] * 3,
        shuffle=True, device=device, sort_key=lambda x: len(x.src))

    encoder = None
    if args.model_type == "transformer":
        config = BertConfig(
            vocab_size=len(text_field.vocab),
            is_decoder=False,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.layers,
            num_attention_heads=args.attention_heads,
            intermediate_size=2 * args.hidden_size,
            hidden_act='relu',
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            num_labels=2)
        encoder = BertModel(config).to(device)
    elif args.model_type == "rnn":
        encoder = RNNEncoder(
            text_field.vocab, args.hidden_size,
            args.hidden_size, args.layers, dropout=0.1)
    elif args.model_type == "embeddings":
        encoder = CNNEncoder(
            text_field.vocab, args.hidden_size,
            args.hidden_size, window=1, layers=0, dropout=0.1)
    else:
        raise ValueError("Uknown uncoder type.")

    model = Stance(encoder, False, 1, [args.hidden_size]).to(device)

    optimizer = optim.Adam(encoder.parameters())

    step = 0
    best_f_score = 0.0
    stalled = 0
    remaining_lr_decrease = args.lr_decrease_count
    learning_rate = args.learning_rate
    threshold = 0.
    for epoch_n in range(args.epochs):
        if remaining_lr_decrease == 0:
            break
        for _, train_batch in enumerate(train_iter):
            if remaining_lr_decrease == 0:
                break
            step += 1

            query = train_batch.src.to(device)
            query_mask = query != 1
            pos = train_batch.tgt.to(device)
            pos_mask = pos != 1
            neg = train_batch.neg.to(device)
            neg_mask = neg != 1

            encoded_query = encoder(query, attention_mask=query_mask)[0]
            encoded_pos = encoder(pos, attention_mask=pos_mask)[0]
            encoded_neg = encoder(neg, attention_mask=neg_mask)[0]

            loss = model.compute_loss(
                encoded_query, query_mask, encoded_pos, pos_mask,
                encoded_neg, neg_mask)

            loss.backward()

            if step % args.delay_update == 0:
                optimizer.step()
                optimizer.zero_grad()
                logging.info(
                    "step: %d (ep. %d), train loss = %.3g",
                    step, epoch_n + 1, loss)

            if step % 10 == 9:
                model.eval()
                (f_score, precision, recall,
                    pos_score_avg, neg_score_avg) = eval_model(
                        encoder, model, val_iter, threshold, device)

                logging.info("")
                logging.info("⊕ %.3f    ⊖ %.3f", pos_score_avg, neg_score_avg)
                logging.info(
                    "P: %.3f   R: %.3f   F: %.3f", precision, recall, f_score)
                threshold = (pos_score_avg + neg_score_avg) / 2

                if f_score > best_f_score:
                    best_f_score = f_score
                    stalled = 0
                    logging.info("New best.")
                else:
                    stalled += 1
                    logging.info("Patience %d/%d.", stalled, args.patience)

                if stalled >= args.patience:
                    learning_rate *= args.lr_decrease_ratio
                    logging.info("Decrease learning rate to %f.", learning_rate)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate
                    remaining_lr_decrease -= 1
                    stalled = 0

                logging.info("")
                model.train()

        logging.info("")
        logging.info("Training done, best F-score %.3f.", best_f_score)
        logging.info("")
        logging.info("TESTING")
        model.eval()
        (f_score, precision, recall,
            pos_score_avg, neg_score_avg) = eval_model(
                encoder, model, val_iter, threshold, device)
        logging.info("Precision: %.3f", precision)
        logging.info("Recall:    %.3f", recall)
        logging.info("F-Score:   %.3f", f_score)

if __name__ == "__main__":
    main()
