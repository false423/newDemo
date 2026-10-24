# -*- coding: utf-8 -*-
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.model_selection
from scipy import io
import imageio
import os
import torch


def flip(data):
    # 水平镜像填充
    left = np.flip(data, axis=1)  # 左右翻转
    right = left
    padded_h = np.concatenate([left, data, right], axis=1)  # [镜像 | 原数据 | 镜像]

    # 垂直镜像填充
    top = np.flip(padded_h, axis=0)  # 上下翻转
    bottom = top
    padded = np.concatenate([top, padded_h, bottom], axis=0)  # [镜像 | 原数据+水平镜像 | 镜像]

    return padded



def ChooseSource(index):
    if index == 0 :
        return ["PaviaU", "Salinas", "PaviaC"]
    elif index == 1 :
        return ["PaviaU", "Salinas", "PaviaC"]

def show_per_num(train_index, test_index):

    class_count = np.max(test_index)

    # 统计训练集和测试集每个类别的数量
    train_num = np.bincount(train_index.ravel(), minlength=class_count + 1)
    test_num = np.bincount(test_index.ravel(), minlength=class_count + 1)

    # 打印训练集和测试集每个类别的数量
    print("Train class counts:", train_num)
    print("Test class counts:", test_num)


def get_device(ordinal):
    # Use GPU
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device


def show_per_num(train_index, test_index):

    class_count = np.max(test_index)

    # 统计训练集和测试集每个类别的数量
    train_num = np.bincount(train_index.ravel(), minlength=class_count + 1)
    test_num = np.bincount(test_index.ravel(), minlength=class_count + 1)

    # 打印训练集和测试集每个类别的数量
    print("Train class counts:", train_num)
    print("Test class counts:", test_num)

def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return imageio.imread(dataset)
    # elif ext == '.hdr':
    #     img = spectral.open_image(dataset)
    #     return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))


def build_dataset(mat, gt, ignored_labels=None):
    """Create a list of training samples based on an image and a mask.

    Args:
        mat: 3D hyperspectral matrix to extract the spectrums from
        gt: 2D ground truth
        ignored_labels (optional): list of classes to ignore, e.g. 0 to remove
        unlabeled pixels
        return_indices (optional): bool set to True to return the indices of
        the chosen samples
        return sample for each class and their count 
    """
    samples = []
    labels = []
    # Check that image and ground truth have the same 2D dimensions
    assert mat.shape[:2] == gt.shape[:2]

    for label in np.unique(gt):
        if label in ignored_labels:
            continue
        else:
            indices = np.nonzero(gt == label)
            samples += list(mat[indices])
            labels += len(indices[0]) * [label]
    return np.asarray(samples), np.asarray(labels)

def get_random_pos(img, window_shape):
    """ Return the corners of a random window in the input image

    Args:
        img: 2D (or more) image, e.g. RGB or grayscale image
        window_shape: (width, height) tuple of the window

    Returns:
        xmin, xmax, ymin, ymax: tuple of the corners of the window

    """
    w, h = window_shape
    W, H = img.shape[:2]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def sample_gt(gt, train_size, mode, values=None):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    
    indices = np.nonzero(gt)
    X = list(zip(*indices)) # x,y features
    y = gt[indices].ravel() # classes
    # train_gt = np.zeros_like(gt)
    # test_gt = np.zeros_like(gt)
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    # train_gt = np.full(train_gt.shape, -1)
    # test_gt = np.full(train_gt.shape, -1)
    
    if train_size > 1:
       train_size = int(train_size)  # int() 函数用于将一个字符串或数字转换为整型。x -- 字符串或数字。 base -- 进制数，默认十进制。
    
    if mode == 'random':
    #    print("当选取样本的时候,Sampling {} with train size = {}".format(mode, train_size))

       train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y)
       print(len(train_indices) + len(test_indices))
       
       #tuple to list
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       #get ramdom sample
       train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
       test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

    elif mode == 'fixed':

    #    print("当选取样本的时候,Sampling {} with train size = {}".format(mode, train_size))
       
       train_indices, test_indices = [], []
    #    train_size = np.ones(class_count + 1) * train_size


       for c in np.unique(gt):
           if c == 0:
               continue
           if values != None:
               if c not in values:
                   continue
               
           indices = np.nonzero(gt == c)
           X = list(zip(*indices))
           
           train, test = sklearn.model_selection.train_test_split(X, train_size=train_size)
           train_indices += train
           test_indices += test
           #*转置
       
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
       test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

    elif mode == 'fixed-100':
        # 选择 固定数量样本
       '''
        sample_gt 函数输出的 train_gt 中各样本数量：  Counter({0: 206500, 1: 100, 4: 100, 2: 100, 8: 100, 5: 100, 9: 100, 6: 100, 3: 100, 7: 100}) 207400
        sample_gt 函数输出的 test_gt 中各样本数量：  Counter({0: 165524, 2: 18549, 1: 6531, 6: 4929, 8: 3582, 4: 2964, 3: 1999, 5: 1245, 7: 1230, 9: 847}) 207400   
       '''
       print("当选取样本的时候，Sampling {} with train size = {}".format(mode, train_size))
       train_indices, test_indices = [], []
       train_size = [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]

       for c in np.unique(gt):
           if c == 0:
              continue  #   continue 语句用来告诉Python跳过当前循环的剩余语句，然后继续进行下一轮循环。
           indices = np.nonzero(gt == c)
           X = list(zip(*indices)) # x,y features
           print(train_size[c],c)

           train, test = sklearn.model_selection.train_test_split(X, train_size=train_size[c])
           train_indices += train
           test_indices += test
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
       test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

    elif mode == 'disjoint':
        print("当选取样本的时候，Sampling {} with train size = {}".format(mode, train_size))
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            #get the coordinate
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                #achieve the request and quit
                try:
                    ratio = first_half_count / second_half_count
                    if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            #divide into train and test set
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt




