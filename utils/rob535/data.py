import os
import random

import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.datasets import CIFAR10

import rob535
from glob import glob
import csv
import numpy as np
from typing import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


classes = np.loadtxt('classes.csv', skiprows=1, dtype=str, delimiter=',')
labels = classes[:, 2].astype(np.uint8)

EECS598 = 0
YOLO = 1


def _extract_tensors(dset, num=None, x_dtype=torch.float32):
    """
    Extract the data and labels from a CIFAR10 dataset object and convert them to
    tensors.

    Input:
    - dset: A torchvision.datasets.CIFAR10 object
    - num: Optional. If provided, the number of samples to keep.
    - x_dtype: Optional. data type of the input image

    Returns:
    - x: `x_dtype` tensor of shape (N, 3, 32, 32)
    - y: int64 tensor of shape (N,)
    """
    x = torch.tensor(dset.data, dtype=x_dtype).permute(0, 3, 1, 2).div_(255)
    y = torch.tensor(dset.targets, dtype=torch.int64)
    if num is not None:
        if num <= 0 or num > x.shape[0]:
            raise ValueError(
                "Invalid value num=%d; must be in the range [0, %d]" % (num, x.shape[0])
            )
        x = x[:num].clone()
        y = y[:num].clone()
    return x, y


def cifar10(num_train=None, num_test=None, x_dtype=torch.float32):
    """
    Return the CIFAR10 dataset, automatically downloading it if necessary.
    This function can also subsample the dataset.

    Inputs:
    - num_train: [Optional] How many samples to keep from the training set.
      If not provided, then keep the entire training set.
    - num_test: [Optional] How many samples to keep from the test set.
      If not provided, then keep the entire test set.
    - x_dtype: [Optional] Data type of the input image

    Returns:
    - x_train: `x_dtype` tensor of shape (num_train, 3, 32, 32)
    - y_train: int64 tensor of shape (num_train, 3, 32, 32)
    - x_test: `x_dtype` tensor of shape (num_test, 3, 32, 32)
    - y_test: int64 tensor of shape (num_test, 3, 32, 32)
    """
    download = not os.path.isdir("cifar-10-batches-py")
    dset_train = CIFAR10(root=".", download=download, train=True)
    dset_test = CIFAR10(root=".", train=False)
    x_train, y_train = _extract_tensors(dset_train, num_train, x_dtype)
    x_test, y_test = _extract_tensors(dset_test, num_test, x_dtype)

    return x_train, y_train, x_test, y_test

def rot(n):
    n = np.asarray(n).flatten()
    assert(n.size == 3)

    theta = np.linalg.norm(n)
    if theta:
        n /= theta
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

        return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    else:
        return np.identity(3)


def get_bbox(p0, p1):
    """
    Input:
    *   p0, p1
        (3)
        Corners of a bounding box represented in the body frame.

    Output:
    *   v
        (3, 8)
        Vertices of the bounding box represented in the body frame.
    *   e
        (2, 14)
        Edges of the bounding box. The first 2 edges indicate the `front` side
        of the box.
    """
    v = np.array([
        [p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
        [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
        [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]
    ])
    e = np.array([
        [2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
        [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]
    ], dtype=np.uint8)

    return v, e

def get_img(file, b, img_size:Tuple[int, int]):
    img_name = np.fromfile(file.replace('_bbox.bin', '_image.jpg'), dtype=np.float32)
    img_name = file.replace('_bbox.bin', '_image.jpg')
    img = plt.imread(img_name)
    R = rot(b[0:3])
    t = b[3:6]
    proj = np.fromfile(file.replace('_bbox.bin', '_proj.bin'), dtype=np.float32)
    proj.resize([3, 4])
    sz = b[6:9]
    vert_3D, edges = get_bbox(-sz / 2, sz / 2)
    vert_3D = R @ vert_3D + t[:, np.newaxis]
    vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
    vert_2D = vert_2D / vert_2D[2, :]

    x_min = int(min(vert_2D[0,:]))
    x_max = int(max(vert_2D[0,:]))
    y_min = int(min(vert_2D[1,:]))
    y_max = int(max(vert_2D[1,:]))
    cropped_img = img[y_min:y_max, x_min:x_max,:]
    resized_img = cv2.resize(cropped_img, img_size)
    return resized_img

def get_2D_bbox(file, b):
    img_name = np.fromfile(file.replace('_bbox.bin', '_image.jpg'), dtype=np.float32)
    img_name = file.replace('_bbox.bin', '_image.jpg')
    img = plt.imread(img_name)
    R = rot(b[0:3])
    t = b[3:6]
    proj = np.fromfile(file.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
    proj.resize([3, 4])
    sz = b[6:9]
    vert_3D, edges = get_bbox(-sz / 2, sz / 2)
    vert_3D = R @ vert_3D + t[:, np.newaxis]
    vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
    vert_2D = vert_2D / vert_2D[2, :]

    h, w, _ = img.shape
    x_min = max(min(vert_2D[0,:]),1) /w
    x_max = min(max(vert_2D[0,:]), w) /w
    y_min = max(min(vert_2D[1,:]), 1) /h
    y_max = min(max(vert_2D[1,:]), h) /h
    # TODO: to get image size
    # print(1)
    rtn = [(x_min + x_max) / 2, (y_min + y_max) / 2, (x_max - x_min), (y_max - y_min)]
    rtn = [float(i) for i in rtn]
    if "06acf647-e2e9-4562-aa42-eed217e5bd84/0058_image" in img_name:
        # print(1)
        pass
    return rtn



def extract_info(
        path,
        img_size: Tuple[int, int],
        train: bool = True,
        bounding_box:bool=False,
        num_train=None,
        num_test=None,
        x_dtype=torch.float32):
    """
    for EECS598 framework
    :param path:
    :param img_size:
    :param num_train:
    :param num_test:
    :param x_dtype:
    """
    print(os.getcwd())

    files = glob('{}/*/*_image.jpg'.format(path))
    files.sort()
    name = '{}/trainval_labels.csv'.format(path)

    N = len(files)
    print("Detected {} training images".format(N))
    X = np.zeros((N, 3, img_size[0], img_size[1]))
    Y = np.zeros((N, 1), dtype=np.int)

    index = 0
    count = 0
    if not bounding_box:

        for file in files:
            # if index == 100:
            #     break
            if index % 100 == 0:
                print("[Progress: file {}, no bounding box file {}]".format(index, count))
            img = plt.imread(file)
            img = cv2.resize(img, img_size)
            X[index] = np.transpose(img, (2, 0, 1))
            if train:
                bbox_name = file.replace('_image.jpg', '_bbox.bin')
                bbox = np.fromfile(bbox_name, dtype=np.float32)
                bbox = bbox.reshape([-1, 11])
                label = 0
                found_valid = False
                for b in bbox:
                    if bool(b[-1]):
                        continue
                    found_valid = True
                    class_id = b[9].astype(np.uint8)
                    label = labels[class_id]
                if found_valid == False:
                    # print(1)
                    count += 1
                Y[index] = label
            index += 1

    files = glob('{}/*/*_bbox.bin'.format(path))
    # has bounding box
    if bounding_box:
        for file in files:
            guid = file.split('/')[-2]
            idx = file.split('/')[-1].replace('_bbox.bin', '')

            bbox = np.fromfile(file, dtype=np.float32)
            bbox = bbox.reshape([-1, 11])
            found_valid = False
            for b in bbox:
                # ignore_in_eval
                if bool(b[-1]):
                    continue
                found_valid = True
                class_id = b[9].astype(np.uint8)
                label = labels[class_id]
                try:
                    img = get_img(file, b, img_size)
                    X[index]  = np.transpose(img, (2, 0, 1))
                    Y[index] = label
                    index += 1

                except Exception as e:
                    print(e)

            if not found_valid:
                label = 0
    return X, Y

def carData(path, img_size:Tuple[int, int], x_dtype):
    train_path = path + "trainval/"
    test_path = path + "test/"
    X_train, Y_train = extract_info(train_path, img_size, train=True)
    # save to npy file
    np.save('X_train.npy', X_train)
    np.save('Y_train.npy', Y_train)
    X_test, Y_test = extract_info(test_path, img_size, train=False)
    np.save('X_test.npy', X_test)
    return X_train, Y_train, X_test, Y_test

def extract_info_YOLO(path:str, train:bool):
    print(os.getcwd())

    files = glob('{}/*/*_image.jpg'.format(path))
    N = len(files)
    print("Detected {} training images".format(N))
    files.sort()

    name = '{}/trainval_labels.csv'.format(path)


    debug_index = 0
    inner_index = 0
    outer_index = 0
    if train:

        for file in files:
            guid = file.split('/')[-2]
            idx = file.split('/')[-1].replace('_bbox.bin', '')
            bbox_name = file.replace('_image.jpg', '_bbox.bin')
            bbox = np.fromfile(bbox_name, dtype=np.float32)
            bbox = bbox.reshape([-1, 11])
            found_valid = False
            label = 0
            outer_index += 1

            for b in bbox:
                inner_index += 1
                # ignore_in_eval
                # if bool(b[-1]):
                #     continue
                found_valid = True
                class_id = b[9].astype(np.uint8)
                label = labels[class_id]
                # try:
                bbox_2D = get_2D_bbox(file, b)
                bbox_2D = ['{:.3f}'.format(x) for x in bbox_2D]

                f = open(file.replace("jpg", "txt"), "w")
                f.write("{} {}".format(int(label), " ".join(bbox_2D)))
                f.close()
                if int(label) == 2:
                    # print(1)
                    pass
                # except Exception as e:
                #     print(e)

            if not found_valid:
                label = 0

            if outer_index % 100 == 0:
                print("[{} iteration, {} inner iteration]".format(outer_index, inner_index))
                if outer_index == 500:
                    # break
                    pass



    else:
        pass
        # test_file = open("{}/test_car.txt".format(path), "w")
        # test_file.write("\n".join(files))
        # test_file.close()
    return files

def generate_data(path:str,framework:int=EECS598, img_size:Tuple[int,int]=(32,32), dtype=torch.float32) -> None:
    # X_train, y_train, X_test, y_test = carData(path, img_size, x_dtype=dtype)
    if framework == EECS598:
        carData(path, img_size, x_dtype=dtype)
    elif framework == YOLO:
        train_file = open("{}/train_car.txt".format(path), "w")
        val_file = open("{}/val_car.txt".format(path), "w")
        test_file = open("{}/test_car.txt".format(path), "w")

        train_files = extract_info_YOLO("{}/trainval".format(path), train=True)
        test_files = extract_info_YOLO("{}/test".format(path), train=False)

        random.shuffle(train_files)
        split_index = int(0.8 * len(train_files))
        train_file.write("\n".join(train_files[:split_index]))
        val_file.write("\n".join(train_files[split_index:]))
        train_file.close()
        val_file.close()

        test_file.write("\n".join(test_files))
        test_file.close()

def preprocess_dataset(
    cuda=True,
    show_examples=True,
    bias_trick=False,
    flatten=True,
    validation_ratio=0.2,
    dtype=torch.float32,
    img_size:Tuple[int,int]=(32,32)
):
    """
    Returns a preprocessed version of the CIFAR10 dataset, automatically
    downloading if necessary. We perform the following steps:

    (0) [Optional] Visualize some images from the dataset
    (1) Normalize the data by subtracting the mean
    (2) Reshape each image of shape (3, 32, 32) into a vector of shape (3072,)
    (3) [Optional] Bias trick: add an extra dimension of ones to the data
    (4) Carve out a validation set from the training set

    Inputs:
    - cuda: If true, move the entire dataset to the GPU
    - validation_ratio: Float in the range (0, 1) giving the fraction of the train
      set to reserve for validation
    - bias_trick: Boolean telling whether or not to apply the bias trick
    - show_examples: Boolean telling whether or not to visualize data samples
    - dtype: Optional, data type of the input image X

    Returns a dictionary with the following keys:
    - 'X_train': `dtype` tensor of shape (N_train, D) giving training images
    - 'X_val': `dtype` tensor of shape (N_val, D) giving val images
    - 'X_test': `dtype` tensor of shape (N_test, D) giving test images
    - 'y_train': int64 tensor of shape (N_train,) giving training labels
    - 'y_val': int64 tensor of shape (N_val,) giving val labels
    - 'y_test': int64 tensor of shape (N_test,) giving test labels

    N_train, N_val, and N_test are the number of examples in the train, val, and
    test sets respectively. The precise values of N_train and N_val are determined
    by the input parameter validation_ratio. D is the dimension of the image data;
    if bias_trick is False, then D = 32 * 32 * 3 = 3072;
    if bias_trick is True then D = 1 + 32 * 32 * 3 = 3073.
    """
    # for path in ['../data/trainval/']
    show_examples = False
    path = "32x32"
    X_train = np.load('X_train.npy')
    y_train = np.load('Y_train.npy')
    X_test = np.load('X_test.npy')

    X_train = torch.tensor(X_train, dtype=dtype)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    X_test = torch.tensor(X_test, dtype=dtype)

    # Move data to the GPU
    if cuda:
        X_train = X_train.cuda()
        y_train = y_train.cuda()
        X_test = X_test.cuda()
        # y_test = y_test.cuda()

    # 0. Visualize some examples from the dataset.
    if show_examples:
        classes = [
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        samples_per_class = 12
        samples = []
        rob535.reset_seed(0)
        for y, cls in enumerate(classes):
            plt.text(-4, 34 * y + 18, cls, ha="right")
            (idxs,) = (y_train == y).nonzero(as_tuple=True)
            for i in range(samples_per_class):
                idx = idxs[random.randrange(idxs.shape[0])].item()
                samples.append(X_train[idx])
        img = torchvision.utils.make_grid(samples, nrow=samples_per_class)
        plt.imshow(rob535.tensor_to_image(img))
        plt.axis("off")
        plt.show()

    # 1. Normalize the data: subtract the mean RGB (zero mean)
    mean_image = X_train.mean(dim=(0, 2, 3), keepdim=True)
    X_train -= mean_image
    X_test -= mean_image

    # 2. Reshape the image data into rows
    if flatten:
      X_train = X_train.reshape(X_train.shape[0], -1)
      X_test = X_test.reshape(X_test.shape[0], -1)

    # 3. Add bias dimension and transform into columns
    if bias_trick:
        ones_train = torch.ones(X_train.shape[0], 1, device=X_train.device)
        X_train = torch.cat([X_train, ones_train], dim=1)
        ones_test = torch.ones(X_test.shape[0], 1, device=X_test.device)
        X_test = torch.cat([X_test, ones_test], dim=1)

    # 4. take the validation set from the training set
    # Note: It should not be taken from the test set
    # For random permumation, you can use torch.randperm or torch.randint
    # But, for this homework, we use slicing instead.
    num_training = int(X_train.shape[0] * (1.0 - validation_ratio))
    num_validation = X_train.shape[0] - num_training

    # return the dataset
    data_dict = {}
    data_dict["X_val"] = X_train[num_training : num_training + num_validation]
    data_dict["y_val"] = y_train[num_training : num_training + num_validation]
    data_dict["X_train"] = X_train[0:num_training]
    data_dict["y_train"] = y_train[0:num_training]

    data_dict["X_test"] = X_test
    # data_dict["y_test"] = y_test
    return data_dict
