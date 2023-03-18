#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -N ddsp_di_all1
#PJM -j
#PJM -m b
#PJM -m e

source /work/01/gk77/k77021/.bashrc
echo "loaded source"
export HOME=/work/01/gk77/k77021
pwd
pip install .


python scripts/train.py \
  --gin-file gin/train/train_newt.gin \
  --dataset-path /work/gk77/k77021/nws/nws-di \
  --checkpoint-path /work/gk77/k77021/nws/nws-di \
  --load-data-to-memory