#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=share-interactive
#PJM -N nws_data_all1
#PJM -j

source /work/01/gk77/k77021/.bashrc
echo "loaded source"
export HOME=/work/01/gk77/k77021
pwd
pip install -e .

echo "====== GPU info ======"
nvidia-smi
echo "======================"


python scripts/create_dataset.py \
--gin-file gin/data/urmp_4second_crepe.gin \
--data-directory /work/gk77/k77021/data/A_sharp_3 \
--output-directory /work/gk77/k77021/nws \
--device cuda:0


#python scripts/train.py \
#  --gin-file gin/train/train_newt.gin \
#  --dataset-path /work/gk77/k77021/nws \
#  --load-data-to-memory