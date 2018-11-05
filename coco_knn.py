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


parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                    choices=['alexnet', 'vgg16'], default='alexnet',
                    help='CNN architecture (default: alexnet)')
parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                    default='Kmeans', help='clustering algorithm (default: Kmeans)')
parser.add_argument('--nmb_cluster', '--k', type=int, default=10000,
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
parser.add_argument('--epochs', type=int, default=200,
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
parser.add_argument('--knn', type=int, default='10', help='k of k-NN')
parser.add_argument('--verbose', action='store_true', help='chatty')


def main():
    global args
    args = parser.parse_args()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    ######################################################################

    print('unpickle clustering objects...')
    handle = open(os.path.join(args.exp, "features.obj"), "rb")
    features = pickle.load(handle)
    handle.close()

    handle = open(os.path.join(args.exp, "train_dataset.obj"), "rb")
    train_dataset = pickle.load(handle)
    handle.close()

    handle = open(os.path.join(args.exp, "images_lists.obj"), "rb")
    images_lists = pickle.load(handle)
    handle.close()

    handle = open(os.path.join(args.exp, "dataset.imgs.obj"), "rb")
    dataset_imgs = pickle.load(handle)
    handle.close()

    #####################################################
    # calculate 10-NN for each feature of 1st cluster
    feature_ids_cluster_0 = images_lists[0]
    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # print('cluster_0: %s' % str(feature_ids_cluster_0))
    # assert len(features) == len(dataset_imgs)
    # for i in feature_ids_cluster_0:
    #     print(i, '---', np.linalg.norm(features[i]), '---', dataset_imgs[i])
    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    d = features.shape[1]  # dimension
    print('features.shape = %s' % (str(features.shape)))
    print('dimension: %d' % d)
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    features_cluster_0 = np.zeros((len(feature_ids_cluster_0), features.shape[1])).astype('float32')
    for j, i in enumerate(feature_ids_cluster_0):
        print(j, '-', i)
        features_cluster_0[j] = features[i]

    print('features_cluster_0.shape = %s' % str(features_cluster_0.shape))
    index.add(features_cluster_0)

    k = args.knn
    print('searching for %d-NN for each feature in cluster...' % k)
    _, I = index.search(features_cluster_0[:1], k + 1)
    print('results: ')
    print(I)
    print('%d NN images for feature %s: ' % (k, str(dataset_imgs[feature_ids_cluster_0[0]])))
    for i in range(k+1):
        print('index into cluster_0: %d' % I[0][i])
        id_into_dataset = feature_ids_cluster_0[I[0][i]]
        print('index into dataset_imgs: %d' % id_into_dataset)
        print(dataset_imgs[id_into_dataset])


    ######################################################################


if __name__ == '__main__':
    main()
