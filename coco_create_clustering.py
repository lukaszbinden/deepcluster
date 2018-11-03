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

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    cudnn.benchmark = True

    # creating cluster assignments log
    # cluster_log = Logger(os.path.join(args.exp, 'clusters'))

    # clustering algorithm to use
    deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster)

    # training convnet with DeepCluster
    for epoch in range(args.start_epoch, args.epochs):
        end = time.time()

        # remove head
        # model.top_layer = None
        # model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        ######################################################################
        root_dir_dataset = 'coco/2017_training/clust_imags/'
        # root_dir_dataset contains a symlink being a pseudolabel and pointing to the
        # folder containing all images of the dataset (this is used by ImageFolderWithPaths)
        toTensor = transforms.Compose([
            transforms.ToTensor()
        ])
        print('ImageFolderWithPaths...')
        dataset = ImageFolderWithPaths(root_dir_dataset, transform=toTensor)
        print('DataLoader...')
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=1,
                                                 num_workers=1,
                                                 pin_memory=True)

        # get the features for the whole dataset
        handle = open("coco/2017_training/filename_feature_dict.obj", "rb")
        filename_feature_dict = pickle.load(handle)
        handle.close()
        print('len(dataset)...............:', len(dataset))
        print('len(filename_feature_dict).:', len(filename_feature_dict))
        print('load_features...')
        features = load_features(dataloader, filename_feature_dict, len(dataset))

        # cluster the features
        print('deepcluster.cluster...')
        clustering_loss = deepcluster.cluster(features, verbose=args.verbose)

        # assign pseudo-labels
        print('clustering.cluster_assign...')
        train_dataset = clustering.cluster_assign(deepcluster.images_lists, dataset.imgs)

        print('save objects...')
        basedir = 'coco_main_out'
        handle = open(os.path.join(basedir, "dataset.imgs.obj"), "wb")
        pickle.dump(dataset.imgs, handle)
        handle.close()
        # >> > dataset_imgs[0]
        # ('coco/2017_training/clust_imags/pseudo_label/000000000009_1.jpg', 0)
        # -> entire dataset as a list of filenames (tuples with path, 0)

        handle = open(os.path.join(basedir, "train_dataset.obj"), "wb")
        pickle.dump(train_dataset, handle)
        handle.close()
        # >> > type(train_dataset)
        # <class 'clustering.ReassignedDataset'>

        handle = open(os.path.join(basedir, "images_lists.obj"), "wb")
        pickle.dump(deepcluster.images_lists, handle)
        handle.close()
        # >> > images_lists[0]
        # [4062, 6195, 9688, 14201, 16309, 21097, 29891, ...]
        # -> list of clusters with indexes into dataset.imgs to get filenames belonging to that cluster

        # cluster_log.log(deepcluster.images_lists)

        ######################################################################


def load_features(dataloader, fs_fns_dict, N):
    for i, (file_tensor, _, filename) in enumerate(dataloader):
        filename = os.path.basename(filename[0])
        feature = fs_fns_dict[filename]

        if i == 0:
            features = np.zeros((N, 512)).astype('float32')

        if i < len(dataloader) - 1:
            features[i * 1: (i + 1) * 1] = feature.astype('float32')
        else:
            # special treatment for final batch
            features[i * 1:] = feature.astype('float32')

    return features


def compute_features(dataloader, model, N):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, _) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
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
