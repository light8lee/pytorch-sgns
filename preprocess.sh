#!/bin/bash
python preprocess.py --data_dir=data \
    --vocab=vocab.txt \
    --corpus=data/origin.txt \
    --window=5 &> preprocess.log
