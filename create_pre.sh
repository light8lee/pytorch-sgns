#!/bin/bash
python create_pretraining_data.py --input_file=../E4G/data/origin.txt \
    --output_file=./data/corpus.txt \
    --vocab_file=vocab.txt