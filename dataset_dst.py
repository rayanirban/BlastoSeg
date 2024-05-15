"""How to read and load the data"""

import os
import imageio
import matplotlib.pyplot as plt
from matplotlib import gridspec, ticker
import numpy as np
from PIL import Image
import torch
import napari
from tifffile import imread, imwrite
from torch.utils.data import Dataset
from torchvision import transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import instance_to_semantic, compute_indiv_sdt

from skimage.segmentation import relabel_sequential
from scipy.optimize import linear_sum_assignment

class BlastoDataset(Dataset):
    """A PyTorch dataset to load membrane labeled images and label masks"""

    def __init__(self, root_dir, transform=None, img_transform=True):
        self.root_dir = root_dir
        self.raw_dir = os.path.join(self.root_dir, 'raw')
        self.label_dir = os.path.join(self.root_dir, 'gt')
        self.dst_dir = os.path.join(self.root_dir, 'gt_dst')
        self.samples = os.listdir(self.raw_dir)  # list the samples
        self.transform = img_transform

        self.raw_image_list = sorted([f for f in os.listdir(self.raw_dir) if '_raw.tif' in f])
        self.label_image_list = sorted([f for f in os.listdir(self.label_dir) if '_gt.tif' in f])
        self.dst_image_list = sorted([f for f in os.listdir(self.dst_dir) if '.tif' in f])

    def img_transform(self, img):
        mean= 117.6194177886187 
        std = 48.85771465056213
        img = (img - mean)/std
        return img 

    # get the total number of samples
    def __len__(self):
        return len(self.raw_image_list)

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # we'll be using Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        image = imread(os.path.join(self.raw_dir, self.raw_image_list[idx]))
        image = image.astype(np.float32)
        image = torch.from_numpy(image.copy())
        image = self.img_transform(image)

        gt = imread(os.path.join(self.label_dir, self.label_image_list[idx]))
        gt = gt.astype(np.int16)
        loss_mask = gt.copy()
        loss_mask = loss_mask != 1
        loss_mask = loss_mask.astype(int)
        loss_mask = torch.from_numpy(loss_mask.copy())

        # compute the distance map per ground truth label
        distmap = imread(os.path.join(self.dst_dir, self.dst_image_list[idx]))
        distmap = torch.from_numpy(distmap.copy())

        return image, distmap, loss_mask
