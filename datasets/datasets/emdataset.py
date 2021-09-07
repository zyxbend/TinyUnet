import os
import collections
import re

# from medicaltorch import transforms as mt_transforms
from torchvision import transforms
from tqdm import tqdm
import numpy as np
# import nibabel as nib
import torch.nn as nn
from torch.utils.data import Dataset
import torch
from torch._six import string_classes, int_classes

from torch.utils.data import DataLoader
from torch import autograd, optim
import torch.backends.cudnn as cudnn

import torchvision.utils as vutils
from PIL import Image

class EM_dataset(Dataset):
    def __init__(self, image_paths, target_paths, train=True):  # initial logic happens like transform
        self.image_paths = image_paths
        self.target_paths = target_paths

        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.target_paths[index])

        image_name = self.image_paths[index].split(".")[0].split("/")[-1]
        t_image, t_mask = self.transforms(image), self.transforms(mask)
        return t_image, t_mask, image_name

    def __len__(self):  # return count of sample we have
        return len(self.image_paths)
