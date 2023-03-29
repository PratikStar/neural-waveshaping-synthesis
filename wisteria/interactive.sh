#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=share-interactive
#PJM -N nws_onenote
#PJM -j

source /work/01/gk77/k77021/.bashrc
echo "loaded source"
export HOME=/work/01/gk77/k77021
pwd
pip install -e .

echo "====== GPU info ======"
nvidia-smi
echo "======================"


#python scripts/create_dataset.py \
#--gin-file gin/data/urmp_4second_crepe.gin \
#--data-directory /work/gk77/k77021/data/timbre_A4 \
#--output-directory /work/gk77/k77021/nws/timbre_A4-16k-f0_hardcoded \
#--device cuda:0


python scripts/train.py \
  --gin-file gin/train/train_newt.gin \
  --dataset-path /work/gk77/k77021/nws/timbre_A4-16k-f0_hardcoded \
  --checkpoint-path /work/gk77/k77021/nws/timbre_A4-16k-f0_hardcoded \
  --load-data-to-memory \
  --restore-checkpoint