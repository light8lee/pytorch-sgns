#!/bin/bash
name=sngs-char-50
mkdir -p outputs/${name}
CUDA_VISIBLE_DEVICES=6 python train.py
    --cuda \
    --data_dir=./data \
	--save_dir=outputs/${name} \
	--epoch=50 \
	--name=${name} \
    --e_dim=300 \
    --weight \
	2> outputs/${name}/${name}.log > outputs/${name}/${name}.info
