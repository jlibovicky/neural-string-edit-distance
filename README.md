# neural-string-edit-distance

This repository contains source code for the paper _Neural String Edit Distance_.

## Sequence classification

Classification using neural edit distance: `train_transliteration_classification.py`

Baseline using Transformers: `train_baseline_cognates_classification.py`

## Sequence generation

Sequence generation using neural string edit distance: `train_transliteration_generation.py`

Baseline using sequence-to-sequence (RNN and Transformer): `train_transliteration_s2s.py`

## Dataset and experiments in the paper

The paper evalutes the method on cognate detection, Arabic-to-English
transliteration and grapheme-to-phoneme conversion. The can be downloaded and
preprocessed for the experiments using the following scripts:

* Cognate detection: `download_cognates.sh`

* Transliteration: `download_transliteration_data.sh`

* Grapheme to phoneme: `download_cmu_dict_data.sh`

## Evaluation using symbol alignment

Interpretability of the models is evaluated by measuring how well symbol
alignment is preserved. The ground-truth data are prepared using a SOTA
statistical aligner. The ground-truth data for alignment can be prepared using
the script `alignment/prepare_alignment.sh` that builds the alignemnt tools.

For extracting the aglignment from the neural string edit distance models, use
the `run_viterbi.py` script with option `--output-format=alignment`.
