#!/bin/bash

if [ -z $VIRTUAL_ENV ]; then
    echo You must be in a virtual environment. > /dev/stderr
    exit 1
fi


if [ ! -e eflomal ]; then
    git clone https://github.com/robertostling/eflomal
    cd eflomal
    make
    make install -e INSTALLDIR=$VIRTUAL_ENV/bin
    pip install cython
    python3 setup.py install
    cd ..
fi

set -ex

for DATA in cmudict transliteration ielex_stat; do
    for TYPE in train eval test; do
        paste <(cut -f1 ../data/${DATA}/${TYPE}.txt | sed -e 's/\(.\)/\1 /g;s/ $//') \
              <(cut -f2 ../data/${DATA}/${TYPE}.txt) | \
        sed -e 's/\t/ ||| /'
    done > ${DATA}.in
    eflomal/align.py --model 3 -i ${DATA}.in -f ${DATA}.fw -r ${DATA}.bw

    ./intersect.py ${DATA}.fw ${DATA}.bw > ${DATA}.align

    tail -n $(wc -l < ../data/${DATA}/test.txt) ${DATA}.align > ${DATA}.align.test
    tail -n $(( `wc -l < ../data/${DATA}/eval.txt` + `wc -l < ../data/${DATA}/test.txt` )) ${DATA}.align | \
        head -n $(wc -l < ../data/${DATA}/eval.txt) > ${DATA}.align.eval
done
