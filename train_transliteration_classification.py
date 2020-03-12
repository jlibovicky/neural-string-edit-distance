#!/usr/bin/env python3

import argparse
import datetime

import torch
from torch import nn, optim
from torch.functional import F
from torchtext import data

from models import EditDistNeuralModelConcurrent


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("data_prefix", type=str)
    args = parser.parse_args()

    ar_text_field = data.Field(
        tokenize=list, init_token="<s>", eos_token="</s>", batch_first=True)
    en_text_field = data.Field(
        tokenize=list, init_token="<s>", eos_token="</s>", batch_first=True)
    labels_field = data.Field(sequential=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, val_data, test_data = data.TabularDataset.splits(
        path=args.data_prefix, train='train.txt',
        validation='eval.txt', test='test.txt', format='tsv',
        fields=[('ar', ar_text_field), ('en', en_text_field),
                ('labels', labels_field)])

    ar_text_field.build_vocab(train_data)
    en_text_field.build_vocab(train_data)
    labels_field.build_vocab(train_data)
    true_class_label = labels_field.vocab.stoi['1']

    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train_data, val_data, test_data), batch_sizes=(32, 32, 32),
        shuffle=True, device=device, sort_key=lambda x: len(x.ar))

    neural_model = EditDistNeuralModelConcurrent(
        ar_text_field.vocab, en_text_field.vocab, device).to(device)

    class_loss = nn.BCELoss()
    kl_div_loss = nn.KLDivLoss(reduction='none')
    xent_loss = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(neural_model.parameters())

    step = 0
    for _ in range(100):
        for i, train_batch in enumerate(train_iter):
            step += 1

            target = (train_batch.labels == true_class_label).float()
            pos_mask = target.unsqueeze(1).unsqueeze(2)
            neg_mask = 1 - pos_mask

            ar_mask = (train_batch.ar != neural_model.ar_pad).float()
            en_mask = (train_batch.en != neural_model.en_pad).float()

            action_mask = ar_mask.unsqueeze(2) * en_mask.unsqueeze(1)

            action_scores, expected_counts, logprobs = neural_model(
                train_batch.ar, train_batch.en)

            bce_loss = class_loss(logprobs.exp(), target)
            pos_samples_loss = kl_div_loss(
                action_scores.reshape(-1, 4),
                expected_counts.reshape(-1, 4)).sum(1)
            neg_samples_loss = xent_loss(
                action_scores.reshape(-1, 4),
                torch.full(action_scores.shape[:-1],
                           3, dtype=torch.long).to(device).reshape(-1))

            pos_loss = (
                (action_mask * pos_mask).reshape(-1) * pos_samples_loss).mean()
            neg_loss = (
                (action_mask * neg_mask).reshape(-1) * neg_samples_loss).mean()
            loss = pos_loss + neg_loss #+ bce_loss

            loss.backward()

            stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            print(f"[{stamp}] step: {step}, train loss = {loss:.3g} "
                  f"(positive: {pos_loss:.3g}, "
                  f"negative: {neg_loss:.3g}, "
                  f"BCE: {bce_loss:.3g})")
            optimizer.step()
            optimizer.zero_grad()

            if step % 50 == 49:
                neural_model.eval()
                with torch.no_grad():
                    neural_false_scores = []
                    neural_true_scores = []
                    for j, val_ex in enumerate(val_iter):
                        neural_score = neural_model.probabilities(val_ex.ar, val_ex.en).exp()
                        if j < 2:
                            for prob, label in zip(neural_score, val_ex.labels):
                                if label == true_class_label:
                                    neural_true_scores.append(prob)
                                else:
                                    neural_false_scores.append(prob)
                        else:
                            print()
                            print(f"neural true  scores: {sum(neural_true_scores) / len(neural_true_scores)}")
                            print(f"neural false scores: {sum(neural_false_scores) / len(neural_false_scores)}")
                            print()
                            break
                neural_model.train()


if __name__ == "__main__":
    main()
