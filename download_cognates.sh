#!/bin/bash

set -ex


#wget https://www.aclweb.org/anthology/attachments/N18-2063.Datasets.zip

# mkdir -p data/ielex
# unzip -p N18-2063.Datasets.zip data/data-ie-42-208.tsv > data/ielex/listing.tsv
# 
# ./classification_data_for_cognates.py data/ielex/listing.tsv > data/ielex/all.tsv
# head -n 20000 data/ielex/all.tsv > data/ielex/eval.txt
# tail -n +20001 data/ielex/all.tsv | head -n 20000 > data/ielex/test.txt
# tail -n +40001 data/ielex/all.tsv > data/ielex/train.txt

mkdir -p data/austro
unzip -p N18-2063.Datasets.zip data/data-aa-58-200.tsv > data/austro/listing.tsv

./classification_data_for_cognates.py data/austro/listing.tsv > data/austro/all.tsv
head -n 20000 data/austro/all.tsv > data/austro/eval.txt
tail -n +20001 data/austro/all.tsv | head -n 20000 > data/austro/test.txt
tail -n +40001 data/austro/all.tsv > data/austro/train.txt
