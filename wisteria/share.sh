#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -N nws_data
#PJM -j
#PJM -m b
#PJM -m e

source /work/01/gk77/k77021/.bashrc
echo "loaded source"
export HOME=/work/01/gk77/k77021
pwd
pip install .

echo "====== GPU info ======"
nvidia-smi
echo "======================"

python scripts/create_dataset.py \
--gin-file gin/data/urmp_4second_crepe.gin \
--data-directory /work/gk77/k77021/data/timbre/monophonic-4secchunks \
--output-directory /work/gk77/k77021/nws/monophonic-4secchunks-di_f0-44032hz \
--device cuda:0


#python scripts/train.py \
#  --gin-file gin/train/train_newt.gin \
#  --dataset-path /work/gk77/k77021/nws/monophonic-4secchunks-di_f0-44k \
#  --checkpoint-path /work/gk77/k77021/nws/monophonic-4secchunks-di_f0-44k \
#  --load-data-to-memory