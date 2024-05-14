"""How to read and load the data"""

import os
import numpy as np
import torch
from tifffile import imread, imwrite
from torch.utils.data import Dataset
from utils import instance_to_semantic

class BlastoDataset(Dataset):
    """A PyTorch dataset to load membrane labeled images and label masks"""

    def __init__(self, root_dir, transform=None, img_transform=True):
        self.root_dir = (
            "/group/dl4miacourse/projects/BlastoSeg/" + root_dir
        )  # the directory with all the training samples

        self.raw_dir = os.path.join(self.root_dir, 'raw')
        self.label_dir = os.path.join(self.root_dir, 'gt')
        self.samples = os.listdir(self.raw_dir)  # list the samples
        self.transform = img_transform

        self.raw_image_list = sorted([f for f in os.listdir(self.raw_dir) if '_raw.tif' in f])
        self.label_image_list = sorted([f for f in os.listdir(self.label_dir) if '_gt.tif' in f])


    def img_transform(self, img):
        mean= 117.6194177886187 
        std = 48.85771465056213
        img = (img - mean)/std
        return img 

    # get the total number of samples
    def __len__(self):
        return len(self.loaded_imgs)

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # we'll be using Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        image = imread(os.path.join(self.raw_dir, self.raw_image_list[idx]))
        image = image.astype(np.float32)
        image = torch.from_numpy(image.copy())
        mask = imread(os.path.join(self.label_dir, self.label_image_list[idx]))
        mask = mask.astype(np.int16)
        mask = instance_to_semantic(mask)
        mask = torch.from_numpy(mask.copy())
        image = self.img_transform(image)

        return image, mask
