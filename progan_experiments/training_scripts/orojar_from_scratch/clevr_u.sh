#!/usr/bin/env bash

python train_orojar.py --hp_lambda 1e-4 --total_kimg 100000 --dataset clevr_simple --nz 3 --num_gpus 4 --warmup_kimg 0
