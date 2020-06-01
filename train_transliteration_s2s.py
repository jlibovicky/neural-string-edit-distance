#!/usr/bin/env python3

"""Sequence generation using standard encoder-decoder models.

The data directory is expected to contains files {train,eval,text}.txt with
tab-separated source and target strings.
"""

import argparse
import datetime
import logging
import os

from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.functional import F
from transformers import BertConfig, BertModel

from experiment import experiment_logging, get_timestamp, save_vocab
from rnn import RNNEncoder, RNNDecoder
from transliteration_utils import (
    load_transliteration_data, decode_ids, char_error_rate)


class Seq2SeqModel(nn.Module):

    def __init__(
            self, encoder, decoder, src_pad_token_id, tgt_bos_token_id,
            tgt_eos_token_id, tgt_pad_token_id, device):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        if isinstance(decoder, RNNDecoder):
            self.transposed_embeddings = decoder.embeddings.weight.t()
        else:
            self.transposed_embeddings = decoder.embeddings.word_embeddings.weight.t()

        self.tgt_bos_token_id = tgt_bos_token_id
        self.tgt_eos_token_id = tgt_eos_token_id
        self.tgt_pad_token_id = tgt_pad_token_id
        self.src_pad_token_id = src_pad_token_id

        self.device = device


    def forward(self, src_batch, tgt_batch):
        encoder_mask = src_batch != self.src_pad_token_id
        encoder_states = self.encoder(
            src_batch,
            attention_mask=encoder_mask)[0]
        decoder_states = self.decoder(
            tgt_batch[:, :-1],
            attention_mask=tgt_batch[:, :-1] != self.tgt_pad_token_id,
            encoder_hidden_states=encoder_states,
            encoder_attention_mask=encoder_mask)[0]
        logits = torch.matmul(
            decoder_states,
            self.transposed_embeddings)
        return logits


    @torch.no_grad()
    def greedy_decode(self, src_batch, max_len=100):
        input_mask = src_batch != self.src_pad_token_id
        encoded, _ = self.encoder(src_batch, attention_mask=input_mask)
        batch_size = encoded.size(0)

        finished = [torch.tensor([False for _ in range(batch_size)]).to(self.device)]
        decoded = [torch.tensor([
            self.tgt_bos_token_id for _ in range(batch_size)]).to(self.device)]

        for _ in range(max_len):
            decoder_input = torch.stack(decoded, dim=1)
            decoder_states, _ = self.decoder(
                decoder_input,
                attention_mask=~torch.stack(finished, dim=1),
                encoder_hidden_states=encoded,
                encoder_attention_mask=input_mask)
            logits = torch.matmul(
                decoder_states,
                self.transposed_embeddings)
            next_symbol = logits[:, -1].argmax(dim=1)
            decoded.append(next_symbol)

            finished_now = next_symbol == self.tgt_eos_token_id + finished[-1]
            finished.append(finished_now)

            if all(finished_now):
                break

        return (torch.stack(decoded, dim=1),
                torch.stack(finished, dim=1).logical_not())

    @torch.no_grad()
    def beam_search(self, src_batch, beam_size=10, max_len=100):
        input_mask = src_batch != self.src_pad_token_id
        encoded, _ = self.encoder(src_batch, attention_mask=input_mask)
        batch_size = encoded.size(0)
        b_range = torch.arange(batch_size)

        cur_len = 1
        current_beam = 1

        decoded = torch.full(
            (batch_size, 1, 1), self.tgt_bos_token_id, dtype=torch.long).to(self.device)
        finished = torch.full(
            (batch_size, 1, 1), False, dtype=torch.bool).to(self.device)
        scores = torch.zeros((batch_size, 1)).to(self.device)

        while cur_len < max_len:
            flat_decoded = decoded.reshape(-1, cur_len)
            flat_finished = finished.reshape(-1, cur_len)

            outputs = self.decoder(
                input_ids=flat_decoded,
                attention_mask=~flat_finished,
                encoder_hidden_states=encoded,
                encoder_attention_mask=input_mask)
            next_token_logprobs = F.log_softmax(torch.matmul(
                outputs[0][:, -1, :],
                self.transposed_embeddings), dim=1)
            vocab_size = next_token_logprobs.size(1)
            past = outputs[1]

            # get scores of all expanded hypotheses
            candidate_scores = (
                scores.unsqueeze(2) +
                next_token_logprobs.reshape(batch_size, current_beam, -1))# / cur_len

            # reshape for beam members and get top k
            best_scores, best_indices = candidate_scores.reshape(
                batch_size, -1).topk(beam_size, dim=-1)
            next_symbol_ids = best_indices % vocab_size
            hypothesis_ids = best_indices // vocab_size

            # numbering elements in the extended batch, i.e. beam size copies of
            # each batch element
            beam_offset = torch.arange(
                0, batch_size * current_beam, step=current_beam,
                dtype=torch.long, device=self.device)
            global_best_indices = (
                beam_offset.unsqueeze(1) + hypothesis_ids).reshape(-1)

            # now select appropriate histories
            decoded = torch.cat((
                flat_decoded.index_select(
                    0, global_best_indices).reshape(batch_size, beam_size, -1),
                next_symbol_ids.unsqueeze(-1)), dim=2)
            reordered_finished = flat_finished.index_select(
                    0, global_best_indices).reshape(batch_size, beam_size, -1)
            finished_now = (
                next_symbol_ids == self.tgt_eos_token_id + reordered_finished[:, :, -1])
            finished = torch.cat((
                reordered_finished,
                finished_now.unsqueeze(-1)), dim=2)

            # re-order scores
            scores = best_scores
            #scores = (
            #    scores.reshape(
            #        -1).index_select(0, global_best_indices).reshape(
            #            batch_size, beam_size) + best_scores)

            # tile encoder after first step
            if cur_len == 1:
                encoded = encoded.unsqueeze(1).repeat(
                    1, beam_size, 1, 1).reshape(
                        batch_size * beam_size, encoded.size(1), encoded.size(2))
                input_mask = input_mask.unsqueeze(1).repeat(1, beam_size, 1).reshape(
                    batch_size * beam_size, -1)

            # in the first iteration, beam size is 1, in the later ones,
            # it is the real beam size
            current_beam = beam_size
            cur_len += 1

        return (decoded[:, 0], finished[:, 0].logical_not())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("data_prefix", type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--model-type", default='transformer',
                        choices=["transformer", "rnn"])
    parser.add_argument("--embedding-dim", default=64, type=int)
    parser.add_argument("--hidden-size", default=128, type=int)
    parser.add_argument("--attention-heads", default=4, type=int)
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument(
        "--src-tokenized", default=False, action="store_true",
        help="If true, source side are space separated tokens.")
    parser.add_argument(
        "--tgt-tokenized", default=False, action="store_true",
        help="If true, target side are space separated tokens.")
    parser.add_argument(
        "--patience", default=10, type=int,
        help="Number of validations witout improvement before finishing.")
    parser.add_argument(
        "--learning-rate", default=1e-4, type=float)
    args = parser.parse_args()

    experiment_params = (
        args.data_prefix.replace("/", "_") +
        f"_{args.model_type}" +
        f"_hidden{args.hidden_size}" +
        f"_attheads{args.attention_heads}" +
        f"_layers{args.layers}" +
        f"_batch{args.batch_size}" +
        f"_lr{args.learning_rate}" +
        f"_patence{args.patience}")
    experiment_dir = experiment_logging(
        f"s2s_{experiment_params}_{get_timestamp()}", args)
    model_path = os.path.join(experiment_dir, "model.pt")
    tb_writer = SummaryWriter(experiment_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ar_text_field, en_text_field, train_iter, val_iter, test_iter = \
        load_transliteration_data(
            args.data_prefix, args.batch_size, device,
            src_tokenized=args.src_tokenized,
            tgt_tokenized=args.tgt_tokenized)

    save_vocab(
        ar_text_field.vocab.itos, os.path.join(experiment_dir, "src_vocab"))
    save_vocab(
        en_text_field.vocab.itos, os.path.join(experiment_dir, "tgt_vocab"))

    if args.model_type == "transformer":
        transformer_config = BertConfig(
            vocab_size=len(ar_text_field.vocab),
            is_decoder=False,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.layers,
            num_attention_heads=args.attention_heads,
            intermediate_size=2 * args.hidden_size,
            hidden_act='relu',
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1)
        encoder = BertModel(transformer_config).to(device)

        transformer_config.is_decoder = True
        transformer_config.vocab_size = len(en_text_field.vocab)
        decoder = BertModel(transformer_config).to(device)
    elif args.model_type == "rnn":
        encoder = RNNEncoder(
            ar_text_field.vocab, args.hidden_size, args.embedding_dim,
            args.layers, dropout=0.1).to(device)
        decoder = RNNDecoder(
            en_text_field.vocab, args.hidden_size, args.embedding_dim,
            args.layers, attention_heads=args.attention_heads,
            dropout=0.1, output_proj=True).to(device)
    else:
        raise RuntimeError("Unknown model type.")

    model = Seq2SeqModel(
        encoder, decoder,
        src_pad_token_id=ar_text_field.vocab.stoi[ar_text_field.pad_token],
        tgt_bos_token_id=en_text_field.vocab.stoi[en_text_field.init_token],
        tgt_eos_token_id=en_text_field.vocab.stoi[en_text_field.eos_token],
        tgt_pad_token_id=en_text_field.vocab.stoi[en_text_field.pad_token],
        device=device)

    nll = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate)

    step = 0
    best_wer = 1e9
    best_wer_step = 0
    best_cer = 1e9
    best_cer_step = 0
    stalled = 0

    for _ in range(args.epochs):
        if stalled > args.patience:
            break
        for train_batch in train_iter:
            if stalled > args.patience:
                break
            step += 1

            logits = model(train_batch.ar, train_batch.en)

            loss = nll(
                logits.reshape([-1, len(en_text_field.vocab)]),
                train_batch.en[:, 1:].reshape([-1]))
            loss.backward()

            if step % 50 == 49:
                logging.info(f"step: {step}, train loss = {loss:.3g}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            if step % 500 == 499:
                tb_writer.add_scalar('train/nll', loss, step)
                model.eval()

                ground_truth = []
                all_hypotheses = []

                for j, val_batch in enumerate(val_iter):
                    with torch.no_grad():
                        src_string = [
                            decode_ids(
                                val_ex, ar_text_field,
                                tokenized=args.src_tokenized)
                            for val_ex in val_batch.ar]
                        tgt_string = [
                            decode_ids(
                                val_ex, en_text_field,
                                tokenized=args.tgt_tokenized)
                            for val_ex in val_batch.en]

                        decoded_val = model.greedy_decode(
                            val_batch.ar,
                            max_len=2 * val_batch.ar.size(1))

                        hypotheses = [
                            decode_ids(out, en_text_field,
                                       tokenized=args.tgt_tokenized)
                            for out in decoded_val[0]]

                        ground_truth.extend(tgt_string)
                        all_hypotheses.extend(hypotheses)

                        if j == 0:
                            for k in range(10):
                                logging.info("")
                                logging.info(
                                    f"'{src_string[k]}' -> "
                                    f"'{hypotheses[k]}' ({tgt_string[k]})")

                logging.info("")
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

                logging.info(f"WER: {wer:.3g}   (best {best_wer:.3g}, "
                      f"step {best_wer_step})")
                logging.info(f"CER: {cer:.3g}   (best {best_cer:.3g}, "
                      f"step {best_cer_step})")
                if stalled > 0:
                    logging.info(f"Stalled {stalled} times.")
                else:
                    torch.save(model, model_path)
                logging.info("")

                tb_writer.add_scalar('val/cer', cer, step)
                tb_writer.add_scalar('val/wer', wer, step)
                tb_writer.flush()
                model.train()

    logging.info("TRAINING FINISHED, evaluating on test data")
    logging.info("")

    model = torch.load(model_path)
    model.eval()

    for j, test_batch in enumerate(test_iter):
        with torch.no_grad():
            src_string = [
                decode_ids(test_ex, ar_text_field,
                           tokenized=args.src_tokenized)
                for test_ex in test_batch.ar]
            tgt_string = [
                decode_ids(test_ex, en_text_field,
                           tokenized=args.tgt_tokenized)
                for test_ex in test_batch.en]

            decoded_val = model.beam_search(
                test_batch.ar,
                max_len=2 * test_batch.ar.size(1))

            hypotheses = [
                decode_ids(out, en_text_field, tokenized=args.tgt_tokenized)
                for out in decoded_val[0]]

            ground_truth.extend(tgt_string)
            all_hypotheses.extend(hypotheses)

    logging.info("")
    wer = 1 - sum(
        float(gt == hyp) for gt, hyp
        in zip(ground_truth, all_hypotheses)) / len(ground_truth)
    logging.info(f"WER: {wer:.3g}")

    cer = char_error_rate(
        all_hypotheses, ground_truth, tokenized=args.tgt_tokenized)
    logging.info(f"CER: {cer:.3g}")
    logging.info("")


if __name__ == "__main__":
    main()
