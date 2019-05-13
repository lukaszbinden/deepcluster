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
import traceback
from util import AverageMeter, Logger, UnifLabelSampler


parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('tile', metavar='TILE', help='which tile to process')
parser.add_argument('knn', metavar='KNN', help='which k to use for k-nn')
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
parser.add_argument('--features', type=str, default='', help='path to precomputed features')
parser.add_argument('--verbose', action='store_true', help='chatty')


def main():
    global args
    args = parser.parse_args()

    print('args: >>>>>>>>>')
    print(args)
    print('<<<<<<<<<<<<<<<')

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch))
    model = models.__dict__[args.arch](sobel=args.sobel)
    model.top_layer = None
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch']
            # remove top_layer parameters from checkpoint
            keys_to_del = []
            for key in checkpoint['state_dict']:
                if 'top_layer' in key:
                    keys_to_del.append(key)
            for key in keys_to_del:
                del checkpoint['state_dict'][key]
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # creating checkpoint repo
    exp_check = os.path.join(args.exp, 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    # preprocessing of data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # tra = [transforms.Resize(224),
    #        transforms.CenterCrop(224),
    #       transforms.ToTensor(),
    #       normalize]
    # cf. encoder_clustering.py: already resized to 224x224
    tra = [transforms.ToTensor(),
           normalize]

    # load the data
    end = time.time()
    tile_name = args.tile
    image_folder = os.path.join(args.data, tile_name)
    print('image folder: %s' % image_folder)
    dataset = datasets.ImageFolder(image_folder, transform=transforms.Compose(tra))
    print('Load dataset: {0:.2f} s'.format(time.time() - end))
    print('len(dataset)...............:', len(dataset))
    print('DataLoader...')
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True)
    print('...DataLoader')

    # clustering algorithm to use
    deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster)

    # remove head
    model.top_layer = None
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

    ######################################################################

    # get the features for the whole dataset
    # if args.features and os.path.exists(args.features):
    #     handle = open(args.features, "rb")
    #     print('load features from filesystem...: %s' % args.features)
    #     features = pickle.load(handle)
    #     print('... loaded.')
    #     handle.close()
    # else:
    print('compute_features...')
    features = compute_features(dataloader, model, len(dataset))
    # if args.features and args.features is not "":
    #     print('dump features to disc...: %s' % args.features)
    #     handle = open(args.features, "wb")
    #     pickle.dump(features, handle)
    #     handle.close()
    #     print('... done')
    print('...compute_features.')

    print('features.shape.:', features.shape)

    # cluster the features
    print('deepcluster.cluster...')
    deepcluster.cluster(features, verbose=args.verbose)

    # assign pseudo-labels
    # print('clustering.cluster_assign...')
    # _ = clustering.cluster_assign(deepcluster.images_lists, dataset.imgs)
    print('number of clusters computed: %d' % len(deepcluster.images_lists))

    # cf. also coco_knn.py
    k = int(args.knn)

    check_small_clusters(deepcluster, k)

    tile_to_10nn = {}
    d = features.shape[1]  # dimension

    for cluster_index, cluster in enumerate(deepcluster.images_lists):
        print('processing cluster %d -->' % (cluster_index + 1))
        start = time.time()
        #####################################################
        # calculate 10-NN for each feature of current cluster
        cluster_feature_ids = cluster

        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        flat_config.device = 0
        index = faiss.GpuIndexFlatL2(res, d, flat_config)

        num_features = len(cluster_feature_ids)
        cluster_features = np.zeros((num_features, features.shape[1])).astype('float32')
        for ind, id in enumerate(cluster_feature_ids):
            # print(ind, '-', id)
            cluster_features[ind] = features[id]

        print('cluster_features.shape = %s' % str(cluster_features.shape))
        index.add(cluster_features)

        l2_knn, knn = index.search(cluster_features, k + 1)  # +1 because 1st is feature itself
        assert knn.shape[0] == cluster_features.shape[0]

        for feature_id in range(num_features):
            for id_nn in range(k + 1):  # id_nn: id of current nearest neighbor
                id_into_dataset = cluster_feature_ids[knn[feature_id][id_nn]]
                img_path = dataset.imgs[id_into_dataset][0]
                name = os.path.basename(img_path).replace('_' + tile_name, '')
                if id_nn == 0:
                    feature_img_name = name
                    knn_list = []
                else:
                    l2_dist = l2_knn[feature_id][id_nn]
                    tuple = (name, l2_dist)
                    knn_list.append(tuple)
            assert len(knn_list) == k
            doAdd = True
            if feature_img_name in tile_to_10nn:
                # special case because of duplicate images in COCO dataset (e.g. 000000000927.jpg und 000000341448.jpg)
                assert knn[feature_id][0] == knn[knn[feature_id][1]][0] \
                       and knn[feature_id][1] == knn[knn[feature_id][1]][1], '\n%d\n%s\n%s\n%s' % (feature_id, str(knn[feature_id]), str(l2_knn[feature_id]), str(knn))
                id_into_dataset = cluster_feature_ids[knn[feature_id][1]]
                img_path = dataset.imgs[id_into_dataset][0]
                name_repl = os.path.basename(img_path).replace('_' + tile_name, '')
                print('duplicate images detected, replacing %s with %s...' % (feature_img_name, name_repl))
                feature_img_name = name_repl
                if feature_img_name in tile_to_10nn:
                    print( '%s already in tile_to_10nn (size: %d, featured_id: %d)' % (feature_img_name, len(tile_to_10nn), feature_id)) 
                    doAdd = False

            if doAdd:
                assert feature_img_name not in tile_to_10nn, '%s already in tile_to_10nn (size: %d, featured_id: %d)' % \
                                                         (feature_img_name, len(tile_to_10nn), feature_id)
                tile_to_10nn[feature_img_name] = knn_list
            else:
                print('skip feature %s altogether...' % feature_img_name)
                doAdd = True

        print(('processing cluster %d <-- [{0:.2f}s]' % (cluster_index + 1)).format(time.time() - start))

    if len(tile_to_10nn) != len(dataset.imgs):
        # assert len(tile_to_10nn) == len(dataset.imgs), '%s vs. %s' % (str(len(tile_to_10nn)), str(len(dataset.imgs)))
        print('len(tile_to_10nn) != len(dataset.imgs): %s vs. %s' % (str(len(tile_to_10nn)), str(len(dataset.imgs))))
        keys = {}
        for img_name in tile_to_10nn.keys():
            keys[img_name] = 1
        for img_path in dataset.imgs:
            name = os.path.basename(img_path[0]).replace('_' + tile_name, '')
            if name in keys:
                del keys[name]
            else:
                print('%s not in tile_to_10nn..' % name)
        print('state of keys after iteration:')
        print(keys)

    out_dir = os.path.join(args.exp, tile_name)
    file_out = os.path.join(out_dir, tile_name + "_" + args.knn + "nn.obj")
    print('pickle map object to \'%s\'...' % file_out)
    handle = open(file_out, "wb")
    pickle.dump(tile_to_10nn, handle)
    handle.close()

    print('done.')
    ########################################################################


def check_small_clusters(deepcluster, k):
    small_clusters = []
    for cluster_index, cluster in enumerate(deepcluster.images_lists):
        if len(cluster) < k + 1:
            small_clusters.append((cluster_index, len(cluster)))
    if len(small_clusters) > 0:
        print('some clusters have fewer points than k = %d neighbors, perhaps decrease nmb_cluster\n clusters: %s'
              % (k, str(small_clusters)))
        assert 1 == 2, 'abort'


def compute_features(dataloader, model, N):
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, _) in enumerate(dataloader):
        with torch.no_grad():
            input_var = torch.autograd.Variable(input_tensor.cuda())
            aux = model(input_var).data.cpu().numpy()

            if i == 0:
                features = np.zeros((N, aux.shape[1])).astype('float32')

            if i < len(dataloader) - 1:
                features[i * args.batch: (i + 1) * args.batch] = aux.astype('float32')
            else:
                # special treatment for final batch
                features[i * args.batch:] = aux.astype('float32')

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.verbose and (i % 200) == 0:
                print('{0} / {1}\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                      .format(i, len(dataloader), batch_time=batch_time))
    return features


if __name__ == '__main__':
    try:
    	main()
    except Exception as e:
        print(e)
        tb = traceback.format_exc()
        print(tb)


