#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python train.py --data_dir=./data \
	--save_dir=./model \
	--epoch=200 \
	--name=sngs-char-200 \
	--cuda 2> sngs-char.log > sngs-char.debug.log
