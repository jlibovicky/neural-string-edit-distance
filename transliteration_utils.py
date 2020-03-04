import re

import editdistance
from torchtext import data

def load_transliteration_data(data_prefix, batch_size, device, src_tokenized=False, tgt_tokenized=False):
    ar_text_field = data.Field(
        tokenize=(lambda s: s.split()) if src_tokenized else list,
        init_token="<s>", eos_token="</s>", batch_first=True)
    en_text_field = data.Field(
        tokenize=(lambda s: s.split()) if tgt_tokenized else list,
        init_token="<s>", eos_token="</s>", batch_first=True)

    train_data, val_data, test_data = data.TabularDataset.splits(
        path=data_prefix, train='train.txt',
        validation='eval.txt', test='test.txt', format='tsv',
        fields=[('ar', ar_text_field), ('en', en_text_field)])

    ar_text_field.build_vocab(train_data)
    en_text_field.build_vocab(train_data)

    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train_data, val_data, test_data),
        batch_sizes=(batch_size, batch_size, batch_size),
        shuffle=True, device=device, sort_key=lambda x: len(x.ar))

    return (ar_text_field, en_text_field, train_iter, val_iter, test_iter)


def decode_ids(ids_list, field, tokenized=False):
    separator = " " if tokenized else ""
    chars = [field.vocab.itos[i] for i in ids_list]
    return separator.join([c for c in chars if c[0] != "<" and c[-1] != ">"])


def char_error_rate(hyps, refs, tokenized=False):
    cers = []
    for hyp, ref in zip(hyps, refs):
        if tokenized:
            hyp, ref = hyp.split(), ref.split()
        edit_ops = editdistance.eval(hyp, ref)
        cers.append(edit_ops / len(ref))
    return sum(cers) / len(cers)

