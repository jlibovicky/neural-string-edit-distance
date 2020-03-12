#!/bin/bash

mkdir -p data/transliteration

curl https://raw.githubusercontent.com/google/transliteration/master/ar2en-train.txt > data/transliteration/train.txt
curl https://raw.githubusercontent.com/google/transliteration/master/ar2en-test.txt > data/transliteration/test.txt
curl https://raw.githubusercontent.com/google/transliteration/master/ar2en-eval.txt > data/transliteration/eval.txt
