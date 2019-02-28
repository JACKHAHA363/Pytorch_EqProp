#!/usr/bin/env bash

source /network/home/luyuchen/miniconda2/etc/profile.d/conda.sh
conda activate py36
export PYTHONUNBUFFERED=1

python ./train_mnist.py

