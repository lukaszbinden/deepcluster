# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DIR="coco/2017_training/version/v6/clustering"
EXP="main_coco_out/cluster_centers/2017_training/version/v5"
KNN="10"
LR=0.05
WD=-5
K=3000
WORKERS=1
BATCH_SIZE=1
PYTHON="/home/lz01a008/.conda/envs/faiss/bin/python -u"
FEATURES="<</home/lz01a008/git/deepcluster/features_v5.obj>>"

mkdir -p ${EXP}



CUDA_VISIBLE_DEVICES=0 ${PYTHON} knowledge_transfer_extract_cluster_centers.py ${DIR} ${TILE} --exp ${EXP} \
    --batch ${BATCH_SIZE} \
    --lr ${LR} --wd ${WD} --k ${K} --features ${FEATURES} --sobel --verbose --workers ${WORKERS} > knowledge_transfer_extract_cluster_centers.log

echo "done."