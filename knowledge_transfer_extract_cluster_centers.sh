# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

LR=0.05
WD=-5
K=2000
WORKERS=1
BATCH_SIZE=1
PYTHON="/home/lz01a008/.conda/envs/faiss/bin/python -u"
FEATURES_RAW="/home/lz01a008/src/logs/20190511_161149/features_raw.obj"
FEATURES_FN="/home/lz01a008/src/logs/20190511_161149/features_filenames.obj"




CUDA_VISIBLE_DEVICES=0 ${PYTHON} knowledge_transfer_extract_cluster_centers.py \
    --batch ${BATCH_SIZE} \
    --lr ${LR} --wd ${WD} --k ${K} --features ${FEATURES_RAW} --features_fn ${FEATURES_FN} --sobel --verbose --workers ${WORKERS} > knowledge_transfer_extract_cluster_centers.log

echo "done."
