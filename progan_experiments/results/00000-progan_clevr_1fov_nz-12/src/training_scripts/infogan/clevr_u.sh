#!/usr/bin/env bash

python train.py --hp_lambda 0 --infogan_lambda 1.0 --infogan_nz 3 --total_kimg 50000 --nz 3 --dataset clevr_simple --num_gpus 4
