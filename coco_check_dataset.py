# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import pickle
import time

import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import clustering
import models
from util import AverageMeter, Logger, UnifLabelSampler


def main():

    # args.data = coco/2017_training/img_root
    root_dir = 'coco/2017_training/clust_imags/'
    toTensor = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = ImageFolderWithPaths(root_dir, transform=toTensor)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             num_workers=1,
                                             pin_memory=True)

    for i, (file_tensor, _, filename) in enumerate(dataloader):
        print(str(i) + ' ---------->')
        print(type(file_tensor))
        print(type(filename))
        print(file_tensor.shape)
        print(filename)
        if i == 3:
            break


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


if __name__ == '__main__':
    main()
