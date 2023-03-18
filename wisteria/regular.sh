#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -N nws_dif0
#PJM -j
#PJM -m b
#PJM -m e

source /work/01/gk77/k77021/.bashrc
echo "loaded source"
export HOME=/work/01/gk77/k77021
pwd
pip install .


python scripts/create_dataset.py \
--gin-file gin/data/urmp_4second_crepe.gin \
--data-directory /work/gk77/k77021/data/timbre/monophonic-4secchunks \
--output-directory /work/gk77/k77021/nws/monophonic-4secchunks-di_f0 \
--device cuda:0

