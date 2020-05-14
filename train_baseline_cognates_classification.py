#!/usr/bin/env python3

import argparse
import datetime
import os

import numpy as np
import torch
from torch import nn, optim
from torch.functional import F
from torchtext import data
from transformers import (
    BertForSequenceClassification, BertConfig,
    get_linear_schedule_with_warmup)

from experiment import experiment_logging, get_timestamp, save_vocab


def cat_examples(dataset, text_field):
    dataset.fields["text"] = text_field
    for example in dataset.examples:
        example.text = example.src + ["<s>", "</s>"] + example.tgt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("data_prefix", type=str)
    parser.add_argument("--hidden-size", default=256, type=int)
    parser.add_argument("--attention-heads", default=8, type=int)
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--delay-update", default=1, type=int,
                        help="Update model every N steps.")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--patience", default=20, type=int,
                        help="Number of validations witout improvement before finishing.")
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    args = parser.parse_args()

    experiment_params = (
        args.data_prefix.replace("/", "_") +
        f"_hidden{args.hidden_size}" +
        f"_attheads{args.attention_heads}" +
        f"_layers{args.layers}" +
        f"_batch{args.batch_size}" +
        f"_patence{args.patience}" +
        f"_delay{args.delay_update}")
    experiment_dir = experiment_logging(
        f"cognates_class_{experiment_params}_{get_timestamp()}", args)
    model_path = os.path.join(experiment_dir, "model.pt")

    text_field = data.Field(
        tokenize=lambda s: s.split(),
        init_token="<s>", eos_token="</s>", batch_first=True)
    labels_field = data.Field(sequential=False, use_vocab=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, val_data, test_data = data.TabularDataset.splits(
        path=args.data_prefix, train='train.txt',
        validation='eval.txt', test='test.txt', format='tsv',
        fields=[('src', text_field), ('tgt', text_field),
                ('label', labels_field)])

    cat_examples(train_data, text_field)
    cat_examples(val_data, text_field)
    cat_examples(test_data, text_field)

    text_field.build_vocab(train_data)
    save_vocab(
        text_field.vocab.itos, os.path.join(experiment_dir, "src_vocab"))
    save_vocab(
        text_field.vocab.itos, os.path.join(experiment_dir, "tgt_vocab"))

    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train_data, val_data, test_data), batch_sizes=[args.batch_size] * 3,
        shuffle=True, device=device, sort_key=lambda x: len(x.text))

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

    model = BertForSequenceClassification(config).to(device)

    optimizer = optim.Adam(model.parameters())

    step = 0
    best_accuracy = 0.0
    for _ in range(args.epochs):
        for i, train_batch in enumerate(train_iter):
            step += 1

            loss, logits = model(
                train_batch.text.to(device),
                labels=train_batch.label.to(device))
            loss.backward()

            stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            print(f"[{stamp}] step: {step}, train loss = {loss:.3g}")
            optimizer.step()
            optimizer.zero_grad()

            if step % 50 == 49:
                model.eval()
                with torch.no_grad():
                    false_scores = []
                    true_scores = []
                    for j, val_ex in enumerate(val_iter):
                        score = F.softmax(model(val_ex.text)[0], dim=1)[:, 1]
                        for prob, label in zip(score, val_ex.label):
                            if label == 1:
                                true_scores.append(prob.cpu().numpy())
                            else:
                                false_scores.append(prob.cpu().numpy())

                        if j > 2:
                            break

                    pos_mean = np.mean(true_scores)
                    neg_mean = np.mean(false_scores)
                    boundary = (pos_mean + neg_mean) / 2
                    accuracy = np.mean(np.concatenate((
                        np.array(true_scores) > boundary, np.array(false_scores) < boundary)))

                    if accuracy > best_accuracy:
                        torch.save(model, model_path)

                    print()
                    print(f"neural true  scores: {pos_mean:.3f} +/- {np.std(true_scores):.3f}")
                    print(f"neural false scores: {neg_mean:.3f} +/- {np.std(false_scores):.3f}")
                    print(f"accuracy: {accuracy:.3f}")
                    print()
                model.train()


if __name__ == "__main__":
    main()
