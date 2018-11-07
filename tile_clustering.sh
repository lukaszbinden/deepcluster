# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DIR="coco/2017_training/clust_imgs_224x224_4285"
EXP="main_coco_out/tile_clustering_4285"
KNN="10"
ARCH="alexnet"
LR=0.05
WD=-5
K=3000
PRE_MODEL="deepcluster_models/alexnet/checkpoint.pth.tar"
WORKERS=1
BATCH_SIZE=1
PYTHON="/home/lz01a008/.conda/envs/faiss/bin/python"

mkdir -p ${EXP}

TILES='t1 t2 t3 t4'

for TILE in ${TILES}
do
    echo "run for "${TILE}"..."
    CUDA_VISIBLE_DEVICES=0 ${PYTHON} tile_clustering.py ${DIR} ${TILE} ${KNN} --exp ${EXP} --arch ${ARCH} \
        --resume ${PRE_MODEL} --batch ${BATCH_SIZE} \
        --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS}
done
echo "done."