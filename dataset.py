"""How to read and load the data"""

import os
import imageio
import matplotlib.pyplot as plt
from matplotlib import gridspec, ticker
import numpy as np
from PIL import Image
import torch
import napari
from torch.utils.data import Dataset
from torchvision import transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage.segmentation import relabel_sequential
from scipy.optimize import linear_sum_assignment


def show_one_image(image_path):
    image = imageio.imread(image_path)
    plt.imshow(image)

class BlastoDataset(Dataset):
    """A PyTorch dataset to load membrane labeled images and label masks"""

    def __init__(self, root_dir, transform=None, img_transform=None):
        self.root_dir = (
            "/group/dl4miacourse/projects/BlastoSeg/" + root_dir
        )  # the directory with all the training samples

        self.raw_dir = os.path.join(self.root_dir, 'raw')
        self.label_dir = os.path.join(self.root_dir, 'gt')
        self.samples = os.listdir(self.raw_dir)  # list the samples

        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )

        raw_image_list = sorted([f for f in os.listdir(self.raw_dir) if '_raw.tif' in f])
        label_image_list = sorted([f for f in os.listdir(self.label_dir) if '_gt.tif' in f])

        self.loaded_imgs = []
        self.loaded_masks = []

        for sample_ind in range(len(self.samples)):
            img_path = os.path.join(
                self.raw_dir, raw_image_list[sample_ind]
            )
            image = imageio.imread(img_path)
            for z in range(image.shape[0]):
                self.loaded_imgs.append(image[z])
                
            mask_path = os.path.join(
                self.label_dir, label_image_list[sample_ind]
            )
            mask = imageio.imread(mask_path)
            for z in range(mask.shape[0]):
                self.loaded_masks.append(mask[z])
        
        # Calculate the mean and std of the intensity over all training slices

        mean_int = np.mean(np.array(self.loaded_imgs))
        std_int = np.std(np.array(self.loaded_imgs))
        print('the mean intensity is', mean_int, 'the std is', std_int)

        self.img_transform = img_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs
        inp_transforms = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize([mean_int], [std_int]),  # 0.5 = mean and 0.5 = variance
            ]
        )

        self.loaded_imgs = [inp_transforms(Image.fromarray(image_slice)) for image_slice in self.loaded_imgs]
        self.loaded_masks = [(transforms.ToTensor()(Image.fromarray(mask_slice))) for mask_slice in self.loaded_masks]

    # get the total number of samples
    def __len__(self):
        return len(self.loaded_imgs)

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # we'll be using Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        image = self.loaded_imgs[idx]
        mask = self.loaded_masks[idx]
        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the image and mask
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
        if self.img_transform is not None:
            image = self.img_transform(image)
        return image, mask

def show_random_dataset_image(dataset):
    idx = np.random.randint(0, len(dataset))  # take a random sample
    img, mask = dataset[idx]  # get the image and the masks
    f, axarr = plt.subplots(1, 2)  # make two plots on one figure
    axarr[0].imshow(img[0])  # show the image
    axarr[0].set_title("Image")
    axarr[1].imshow(mask[0], interpolation=None)  # show the masks
    axarr[1].set_title("Mask")
    _ = [ax.axis("off") for ax in axarr]  # remove the axes
    print("Image size is %s" % {img[0].shape})
    plt.show()

def show_random_dataset_napari(dataset, viewer: napari.Viewer): 
    idx = np.random.randint(0, len(dataset))  # take a random sample
    img, mask = dataset[idx]  # get the image and the nuclei masks
    img.shape
    viewer.add_image(img.numpy())
    viewer.add_labels(np.array(mask, dtype = np.uint16))

def show_random_dataset_image_with_prediction(dataset, model, device="cpu"):
    idx = np.random.randint(0, len(dataset))  # take a random sample
    img, mask = dataset[idx]  # get the image and the nuclei masks
    x = img.to(device).unsqueeze(0)
    y = model(x)[0].detach().cpu().numpy()
    print("MSE loss:", np.mean((mask[0].numpy() - y[0]) ** 2))
    f, axarr = plt.subplots(1, 3)  # make two plots on one figure
    axarr[0].imshow(img[0])  # show the image
    axarr[0].set_title("Image")
    axarr[1].imshow(mask[0], interpolation=None)  # show the masks
    axarr[1].set_title("Mask")
    axarr[2].imshow(y[0], interpolation=None)  # show the prediction
    axarr[2].set_title("Prediction")
    _ = [ax.axis("off") for ax in axarr]  # remove the axes
    print("Image size is %s" % {img[0].shape})
    plt.show()