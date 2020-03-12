#!/bin/bash

mkdir -p data/cmudict

curl https://raw.githubusercontent.com/microsoft/CNTK/master/Examples/SequenceToSequence/CMUDict/Data/cmudict-0.7b.train-dev-20-21 | sed -e 's/  /\t/' > data/cmudict/train.txt
curl https://raw.githubusercontent.com/microsoft/CNTK/master/Examples/SequenceToSequence/CMUDict/Data/cmudict-0.7b.train-dev-1-21 | sed -e 's/  /\t/' > data/cmudict/eval.txt
curl https://raw.githubusercontent.com/microsoft/CNTK/master/Examples/SequenceToSequence/CMUDict/Data/cmudict-0.7b.test | sed -e 's/  /\t/' > data/cmudict/test.txt
