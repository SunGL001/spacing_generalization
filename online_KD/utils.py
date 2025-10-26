import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


import os, glob
from torchvision.io import read_image, ImageReadMode


# class TrainTinyImageNetDataset(Dataset):
#     def __init__(self, id, transform=None):
#         self.filenames = glob.glob("/data/datasets/tiny-imagenet-200/train/*/*/*.JPEG")
#         self.transform = transform
#         self.id_dict = id

#     def __len__(self):
#         return len(self.filenames)

#     def __getitem__(self, idx):
#         img_path = self.filenames[idx]
#         image = read_image(img_path)
#         if image.shape[0] == 1:
#           image = read_image(img_path,ImageReadMode.RGB)
#         label = self.id_dict[img_path.split('tiny-imagenet-200/train')[1].split('/')[1]]
        
#         if self.transform:
#             image = self.transform(image.type(torch.FloatTensor))
#             # image = self.transform(image.float())
#         return image, label
# class TestTinyImageNetDataset(Dataset):
#     def __init__(self, id, transform=None):
#         self.filenames = glob.glob("/data/datasets/tiny-imagenet-200/val/images/*.JPEG")
#         self.transform = transform
#         self.id_dict = id
#         self.cls_dic = {}
#         for i, line in enumerate(open('/data/datasets/tiny-imagenet-200/val/val_annotations.txt', 'r')):
#             a = line.split('\t')
#             img, cls_id = a[0],a[1]
#             self.cls_dic[img] = self.id_dict[cls_id]
 
#     def __len__(self):
#         return len(self.filenames)

#     def __getitem__(self, idx):
#         img_path = self.filenames[idx]
#         image = read_image(img_path)
#         if image.shape[0] == 1:
#           image = read_image(img_path,ImageReadMode.RGB)
#         label = self.cls_dic[img_path.split('/')[-1]]
#         if self.transform:
#             image = self.transform(image.type(torch.FloatTensor))
#             # image = self.transform(image.float())
#         return image, label

from typing import Any
from PIL import Image

class TrainTinyImageNet(Dataset):
    def __init__(self,root, id, transform=None) -> None:
        super().__init__()
        # self.filenames = glob.glob(root + "\\train\*\*\*.JPEG")
        self.filenames = glob.glob(root + "/train/*/*/*.JPEG")
        self.transform = transform
        self.id_dict = id
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx: Any) -> Any:
        img_path = self.filenames[idx]
        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        label = self.id_dict[img_path.split('/')[-3]]
        if self.transform:
            image = self.transform(image)
        return image, label
 
class ValTinyImageNet(Dataset):
    def __init__(self, root, id, transform=None):
        # root='/data/datasets/tiny-imagenet-200'
        # self.filenames = glob.glob(root + "\\val\images\*.JPEG")
        self.filenames = glob.glob(root + "/val/images/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open(root + '/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0], a[1]
            self.cls_dic[img] = self.id_dict[cls_id]
 
    def __len__(self):
        return len(self.filenames)
 
    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image)
        return image, label
 
def load_tinyimagenet(root, batch_size, num_workers,split,shuffle, id_dic):
    
    root = '/data/datasets/tiny-imagenet-200'

    if split=='train':
        transform =transforms.Compose([transforms.Resize(64),
                                     transforms.RandomCrop(64, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        dataset = TrainTinyImageNet(root=root, id=id_dic, transform=transform)
        loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               pin_memory=True,
                                               num_workers=num_workers)
        
    elif split=='val':
        transform=transforms.Compose([transforms.Resize(64),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        dataset = ValTinyImageNet(root=root, id=id_dic, transform=transform)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             pin_memory=True,
                                             num_workers=num_workers)
    return loader



def get_training_dataloader(dataset, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    if dataset == 'cifar100':
        transform_train = transforms.Compose([
            # transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
        data = torchvision.datasets.CIFAR100(root='/data/datasets/CIFAR', train=True, download=True, transform=transform_train)
        loader = DataLoader(data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
    elif dataset == 'tiny_imagenet':
        id_dic = {}
        root = '/data/datasets/tiny-imagenet-200'
        for i, line in enumerate(open(root+'/wnids.txt','r')):
            id_dic[line.replace('\n', '')] = i
        # num_classes = len(id_dic)
        loader = load_tinyimagenet(root=root, batch_size=batch_size, num_workers=num_workers, split='train', shuffle=shuffle, id_dic=id_dic)
    
    elif dataset == 'imagenet':
        transform_train = transforms.Compose([
            # transforms.Resize(224),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        data = torchvision.datasets.ImageNet(root='/data/datasets/ImageNet', split='train', transform=transform_train)
        loader = DataLoader(data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
    return loader

def get_test_dataloader(dataset, batch_size=16, num_workers=4, shuffle=False):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """
    if dataset == 'cifar100':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
        data = torchvision.datasets.CIFAR100(root='/data/datasets/CIFAR', train=False, download=True, transform=transform_test)
        loader = DataLoader(data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    elif dataset == 'tiny_imagenet':
        id_dic = {}
        root = '/data/datasets/tiny-imagenet-200'
        for i, line in enumerate(open(root+'/wnids.txt','r')):
            id_dic[line.replace('\n', '')] = i
        loader = load_tinyimagenet(root=root, batch_size=batch_size, num_workers=num_workers, split='val', shuffle=shuffle, id_dic=id_dic)


    elif dataset == 'imagenet':
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        data = torchvision.datasets.ImageNet(root='/data/datasets/ImageNet', split='val', transform=transform_test)
        loader = DataLoader(data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
    return loader 

def calculate_mean_std(dataset):
    """
    计算数据集的均值和方差
    :param dataset: 数据集
    :return: mean, std
    """
    # init mean and std
    mean = 0.
    std = 0.
    total_samples = len(dataset)
    print("tran dataset len:", total_samples)
    
    # Traversing the dataset calculates the mean and std
    for data, _ in dataset:
        """
        data张量是一个代表图像的张量，通常具有三个维度：(通道, 高度, 宽度)。
        dim=(1, 2)参数指定了要在高度和宽度维度上进行求均值的操作。
        计算每个通道上的像素值的平均值，得到的结果是一个包含每个通道上的平均值的张量。
        """
        data=data.type(torch.float32)
        mean += torch.mean(data, dim=(1, 2))
        std += torch.std(data, dim=(1, 2))
    
    # Calculate the population mean and std
    mean /= total_samples
    std /= total_samples
    
    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]