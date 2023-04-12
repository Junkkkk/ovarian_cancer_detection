import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import random


def data_split(path, split_ratio=0.8):

    train_info = []
    val_info = []
    n_files = os.listdir(os.path.join(path, "n_patch"))
    p_files = os.listdir(os.path.join(path, "p_patch"))
    random.shuffle(n_files)
    random.shuffle(p_files)
    for idx, info in enumerate(n_files):
        if idx < len(n_files) * 0.8:
            train_info.append((os.path.join(path, "n_patch", info), 0))
        else:
            val_info.append((os.path.join(path, "n_patch", info), 0))
    for idx, info in enumerate(p_files):
        if idx < len(p_files) * 0.8:
            train_info.append((os.path.join(path, "p_patch", info), 1))
        else:
            val_info.append((os.path.join(path, "p_patch", info), 1))
    return train_info, val_info


def data_load(path, under_sampling=False):
    data_info = []
    n_files = os.listdir(os.path.join(path, "n_patch"))
    p_files = os.listdir(os.path.join(path, "p_patch"))

    if under_sampling==True:
        random.seed(1)
        n_files = random.sample(n_files, k=len(p_files))
        
    for idx, info in enumerate(n_files):
        data_info.append((os.path.join(path, "n_patch", info), 0))
    for idx, info in enumerate(p_files):
        data_info.append((os.path.join(path, "p_patch", info), 1))
    return data_info


class custom_dataset(torch.utils.data.Dataset):
    def __init__(self, info_list, transforms=None):
        self.info_list = info_list
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(self.info_list[index][0])
        target = self.info_list[index][1]
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.info_list)
