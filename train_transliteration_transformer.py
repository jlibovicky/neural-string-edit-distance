#!/usr/bin/env python3

import argparse
import datetime

from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.functional import F
from transformers import BertConfig, BertModel, Model2Model

from transliteration_utils import (
    load_transliteration_data, decode_ids, char_error_rate)


def greedy_decode(encoder, decoder, src_batch, bos_token_id, eos_token_id, pad_token_id, device, max_len=100):
    input_mask = src_batch != pad_token_id
    encoded, _ = encoder(src_batch)
    batch_size = encoded.size(0)

    finished = [torch.tensor([False for _ in range(batch_size)]).to(device)]
    decoded = [torch.tensor([
        bos_token_id for _ in range(batch_size)]).to(device)]

    for _ in range(max_len):
        decoder_input = torch.stack(decoded, dim=1)
        decoder_states, _ = decoder(
            decoder_input,
            encoder_hidden_states=encoded,
            encoder_attention_mask=input_mask)
        logits = torch.matmul(
            decoder_states,
            decoder.embeddings.word_embeddings.weight.t())
        next_symbol = logits[:, -1].argmax(dim=1)
        decoded.append(next_symbol)

        finished_now = next_symbol == eos_token_id + finished[-1]
        finished.append(finished_now)

        if all(finished_now):
            break

    return (torch.stack(decoded, dim=1),
            torch.stack(finished, dim=1).logical_not())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("data_prefix", type=str)
    parser.add_argument("--hidden-size", default=64, type=int)
    parser.add_argument("--attention-heads", default=4, type=int)
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--src-tokenized", default=False, action="store_true",
                        help="If true, source side are space separated tokens.")
    parser.add_argument("--tgt-tokenized", default=False, action="store_true",
                        help="If true, target side are space separated tokens.")
    parser.add_argument("--patience", default=10, type=int,
                        help="Number of validations witout improvement before finishing.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ar_text_field, en_text_field, train_iter, val_iter, test_iter = \
        load_transliteration_data(
            args.data_prefix, args.batch_size, device,
            src_tokenized=args.src_tokenized,
            tgt_tokenized=args.tgt_tokenized)

    transformer_config = BertConfig(
        vocab_size=len(ar_text_field.vocab),
        is_decoder=False,
        hidden_size=args.hidden_size,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=2 * args.hidden_size,
        hidden_act='relu',
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1)
    encoder = BertModel(transformer_config)

    transformer_config.is_decoder = True
    transformer_config.vocab_size = len(en_text_field.vocab)
    decoder = BertModel(transformer_config)

    seq2seq = Model2Model(encoder, decoder).to(device)

    nll = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(seq2seq.parameters())

    en_bos_token_id = en_text_field.vocab.stoi[en_text_field.init_token]
    en_eos_token_id = en_text_field.vocab.stoi[en_text_field.eos_token]
    en_pad_token_id = en_text_field.vocab.stoi[en_text_field.pad_token]
    ar_pad_token_id = ar_text_field.vocab.stoi[ar_text_field.pad_token]

    en_examples = []
    ar_examples = []
    step = 0
    best_wer = 1.0
    best_wer_step = 0
    best_cer = 1.0
    best_cer_step = 0
    stalled = 0

    stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    experiment_params = (
        args.data_prefix.replace("/", "_") +
        f"_hidden{args.hidden_size}" +
        f"_attheads{args.attention_heads}" +
        f"_layers{args.layers}" +
        f"_batch{args.batch_size}" +
        f"_patence{args.patience}")
    tb_writer = SummaryWriter(f"runs/transformer_{experiment_params}_{stamp}")

    for _ in range(100):
        if stalled > args.patience:
            break
        for i, train_batch in enumerate(train_iter):
            if stalled > args.patience:
                break
            step += 1

            encoder_mask = train_batch.ar != ar_pad_token_id
            encoder_states = encoder(
                train_batch.ar,
                attention_mask=encoder_mask)[0]
            decoder_states = decoder(
                train_batch.en[:, :-1],
                attention_mask=train_batch.en[:, :-1] != ar_pad_token_id,
                encoder_hidden_states=encoder_states,
                encoder_attention_mask=encoder_mask)[0]
            logits = torch.matmul(
                decoder_states,
                decoder.embeddings.word_embeddings.weight.t())

            loss = nll(
                logits.reshape([-1, len(en_text_field.vocab)]),
                train_batch.en[:, 1:].reshape([-1]))
            loss.backward()

            if step % 50 == 49:
                print(f"step: {step}, train loss = {loss:.3g}")
            torch.nn.utils.clip_grad_norm_(seq2seq.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            if step % 500 == 499:
                tb_writer.add_scalar('train/nll', loss, step)
                seq2seq.eval()

                ground_truth = []
                all_hypotheses = []

                for j, val_batch in enumerate(val_iter):
                    with torch.no_grad():
                        src_string = [
                            decode_ids(val_ex, ar_text_field, tokenized=args.src_tokenized)
                            for val_ex in val_batch.ar]
                        tgt_string = [
                            decode_ids(val_ex, en_text_field, tokenized=args.tgt_tokenized)
                            for val_ex in val_batch.en]

                        decoded_val = greedy_decode(
                            encoder, decoder, val_batch.ar,
                            en_bos_token_id, en_eos_token_id,
                            en_pad_token_id,
                            device,
                            max_len=2 * val_batch.ar.size(1))

                        hypotheses = [
                            decode_ids(out, en_text_field, tokenized=args.tgt_tokenized)
                            for out in decoded_val[0]]

                        ground_truth.extend(tgt_string)
                        all_hypotheses.extend(hypotheses)

                        if j == 1:
                            for k in range(10):
                                print()
                                print(f"'{src_string[k]}' -> "
                                      f"'{hypotheses[k]}' ({tgt_string[k]})")

                print()
                wer = 1 - sum(
                    float(gt == hyp) for gt, hyp
                    in zip(ground_truth, all_hypotheses)) / len(ground_truth)
                cer = char_error_rate(all_hypotheses, ground_truth, tokenized=args.tgt_tokenized)

                stalled += 1
                if wer < best_wer:
                    best_wer = wer
                    best_wer_step = step
                    stalled = 0
                if cer < best_cer:
                    best_cer = cer
                    best_cer_step = step
                    stalled = 0

                print(f"WER: {wer:.3g}   (best {best_wer:.3g}, step {best_wer_step})")
                print(f"CER: {cer:.3g}   (best {best_cer:.3g}, step {best_cer_step})")
                if stalled > 0:
                    print(f"Stalled {stalled} times.")
                print()

                tb_writer.add_scalar('val/cer', cer, step)
                tb_writer.add_scalar('val/wer', wer, step)
                tb_writer.flush()
                seq2seq.train()

    print("TRAINING FINISHED, evaluating on test data")
    print()

    seq2seq.eval()
    for j, test_batch in enumerate(test_iter):
        with torch.no_grad():
            src_string = [
                decode_ids(test_ex, ar_text_field, tokenized=args.src_tokenized)
                for test_ex in test_batch.ar]
            tgt_string = [
                decode_ids(test_ex, en_text_field, tokenized=args.tgt_tokenized)
                for test_ex in test_batch.en]

            decoded_val = greedy_decode(
                encoder, decoder, test_batch.ar,
                en_bos_token_id, en_eos_token_id,
                en_pad_token_id,
                device,
                max_len=2 * test_batch.ar.size(1))

            hypotheses = [
                decode_ids(out, en_text_field, tokenized=args.tgt_tokenized)
                for out in decoded_val[0]]

            ground_truth.extend(tgt_string)
            all_hypotheses.extend(hypotheses)

    print()
    wer = 1 - sum(
        float(gt == hyp) for gt, hyp
        in zip(ground_truth, all_hypotheses)) / len(ground_truth)

    cer = char_error_rate(all_hypotheses, ground_truth, tokenized=args.tgt_tokenized)
    print(f"WER: {wer:.3g}")
    print(f"CER: {cer:.3g}")


if __name__ == "__main__":
    main()
