# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

TILE="t1"
DIR="coco/2017_training/clust_imgs_224x224"
EXP="main_coco_out/tile_clustering"
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

CUDA_VISIBLE_DEVICES=0 ${PYTHON} main_for_coco.py ${DIR} ${TILE} ${KNN} --exp ${EXP} --arch ${ARCH} \
  --resume ${PRE_MODEL} --batch ${BATCH_SIZE} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS}