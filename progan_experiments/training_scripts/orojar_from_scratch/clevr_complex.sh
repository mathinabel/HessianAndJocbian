#!/usr/bin/env bash

python train_orojar.py --hp_lambda 1e-5 --total_kimg 100000 --dataset clevr_complex --num_gpus 4 --warmup_kimg 0
