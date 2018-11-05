# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DIR="coco/2017_training/clust_imgs_224x224"
EXP="main_coco_out"
PYTHON="/home/lz01a008/.conda/envs/faiss/bin/python"

CUDA_VISIBLE_DEVICES=0 ${PYTHON} coco_knn.py ${DIR} --exp ${EXP} --verbose
