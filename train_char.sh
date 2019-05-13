#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python train.py --data_dir=./data \
	--save_dir=./model50 \
	--epoch=50 \
	--name=sngs-char-50 \
	--cuda 2> sngs-char-50.log > sngs-char-50.debug.log
