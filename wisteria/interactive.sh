#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=share-interactive
#PJM -N nws_data_all
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
--data-directory /work/gk77/k77021/data/timbre/monophonic-4secchunks \
--output-directory /work/gk77/k77021/nws/monophonic-4secchunks \
--device cuda:0

