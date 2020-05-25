#!/bin/bash

set -ex

mkdir -p data/ielex

wget https://www.aclweb.org/anthology/attachments/N18-2063.Datasets.zip

unzip -p N18-2063.Datasets.zip data/data-ie-42-208.tsv > data/ielex/listing.tsv
./classification_data_for_cognates.py data/ielex/listing.tsv > data/ielex/all.tsv
head -n 20000 data/ielex/all.tsv > data/ielex/eval.txt
tail -n +20001 data/ielex/all.tsv | head -n 20000 > data/ielex/test.txt
tail -n +40001 data/ielex/all.tsv > data/ielex/train.txt
