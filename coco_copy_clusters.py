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

import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os.path
import clustering
from util import AverageMeter, Logger

from scipy.misc import imsave

from shutil import copyfile


parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

parser.add_argument('--data', metavar='DIR', default='.', help='path to dataset')
parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                    choices=['alexnet', 'vgg16'], default='alexnet',
                    help='CNN architecture (default: alexnet)')
parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                    default='Kmeans', help='clustering algorithm (default: Kmeans)')
parser.add_argument('--nmb_cluster', '--k', type=int, default=3000,
                    help='number of cluster for k-means (default: 10000)')
parser.add_argument('--lr', default=0.05, type=float,
                    help='learning rate (default: 0.05)')
parser.add_argument('--wd', default=-5, type=float,
                    help='weight decay pow (default: -5)')
parser.add_argument('--reassign', type=float, default=1.,
                    help="""how many epochs of training between two consecutive
                    reassignments of clusters (default: 1)""")
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', type=int, default=1,
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts) (default: 0)')
parser.add_argument('--batch', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: None)')
parser.add_argument('--checkpoints', type=int, default=25000,
                    help='how many iterations between two checkpoints (default: 25000)')
parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
parser.add_argument('--exp', type=str, default='', help='path to exp folder')
parser.add_argument('--verbose', action='store_true', help='chatty')


def main():
    global args
    args = parser.parse_args()

    handle = open("main_coco_out/train_dataset.obj", "rb")
    train_dataset = pickle.load(handle)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
    )

    num_clusters_to_copy = 2
    basedir = 'main_coco_out/clusters'
    prev_id = -1
    for i, (path, target) in enumerate(train_dataloader):
        cluster_id = target.data.cpu().numpy()[0]
        if cluster_id >= num_clusters_to_copy:
            break
        if prev_id != cluster_id:
            prev_id = cluster_id
            dir = os.path.join(basedir, str(cluster_id))
            if not os.path.exists(dir):
                os.makedirs(dir)

        source = path[0]
        filename = os.path.basename(path[0])
        dest = os.path.join(dir, filename)
        copyfile(source, dest)
        # output_tensor = input_tensor.data.cpu().numpy()[0]
        # imsave(name, output_tensor)

        # print(str(i) + ' ---------->')
        # print(output_tensor.shape)
        # print(cluster_id)
        # print(path)
        # print(dest)
        # if i == 3:
        #     break

    ######################################################################


if __name__ == '__main__':
    main()
