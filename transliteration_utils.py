from collections import defaultdict

import editdistance
from torchtext import data


def load_vocab(file):
    vocab = []
    for token in file:
        vocab.append(token.strip())
    file.close()
    stoi = defaultdict(int)
    for i, symb in enumerate(vocab):
        stoi[symb] = i
    return vocab, stoi


def load_transliteration_data(
        data_prefix, batch_size, device, src_tokenized=False,
        tgt_tokenized=False):
    src_text_field = data.Field(
        tokenize=(lambda s: s.split()) if src_tokenized else list,
        init_token="<s>", eos_token="</s>", batch_first=True)
    tgt_text_field = data.Field(
        tokenize=(lambda s: s.split()) if tgt_tokenized else list,
        init_token="<s>", eos_token="</s>", batch_first=True)

    train_data, val_data, test_data = data.TabularDataset.splits(
        path=data_prefix, train='train.txt',
        validation='eval.txt', test='test.txt', format='tsv',
        fields=[('ar', src_text_field), ('en', tgt_text_field)])

    src_text_field.build_vocab(train_data)
    tgt_text_field.build_vocab(train_data)

    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train_data, val_data, test_data),
        batch_sizes=(batch_size, 16, 16),
        shuffle=True, device=device, sort_key=lambda x: len(x.ar))

    return (src_text_field, tgt_text_field, train_iter, val_iter, test_iter)


def decode_ids(ids_list, vocab, tokenized=False):
    if not isinstance(vocab, list):
        vocab = vocab.vocab.itos
    separator = " " if tokenized else ""
    chars = [vocab[i] for i in ids_list]
    if chars[0] == "<s>":
        chars = chars[1:]
    real_chars = []
    for char in chars:
        if char in ["<s>", "</s>", "<pad>"]:
            break
        real_chars.append(char)
    return separator.join(real_chars)


def char_error_rate(hyps, refs, tokenized=False, average=True):
    cers = []
    for hyp, ref in zip(hyps, refs):
        if tokenized:
            hyp, ref = hyp.split(), ref.split()
        edit_ops = editdistance.eval(hyp, ref)
        if not ref:
            # TODO this is a BUG and needs to be fixed
            continue
        cers.append(edit_ops / len(ref))
    if average:
        return sum(cers) / len(cers)
    return cers
