
from skimage.segmentation import find_boundaries, watershed
import numpy as np
from scipy.ndimage import label, maximum_filter, distance_transform_edt
import os 
import tifffile
from skimage.io import imread 

def compute_indiv_sdt(pred: np.ndarray):
    """Function to compute a distance transform per label. Calculate local distance maps and weigh"""

    # compute the distance transform inside and outside of the objects
    dstmap = np.zeros(pred.shape, dtype=np.float32)

    for id in np.unique(pred):
        if id != 0:
            dst = distance_transform_edt(pred == id)
            dstmap += (dst/np.max(dst))

    # be sure to return your solution as type 'float'
    return dstmap.astype(float)

inputdir = '/group/dl4miacourse/projects/BlastoSeg/training/gt'
outputdir = '/group/dl4miacourse/projects/BlastoSeg/training/gt_dst'

if not os.path.exists(outputdir):
    os.mkdir(outputdir)

files = [f for f in os.listdir(inputdir) if '.tif' in f]
for f in files:
    print('processing file', f)
    img = imread(os.path.join(inputdir, f))
    dst = compute_indiv_sdt(img)
    tifffile.imwrite(os.path.join(outputdir, f), np.array(dst, dtype=np.float32))

inputdir = '/group/dl4miacourse/projects/BlastoSeg/validation/gt'
outputdir = '/group/dl4miacourse/projects/BlastoSeg/validation/gt_dst'

if not os.path.exists(outputdir):
    os.mkdir(outputdir)
    
files = [f for f in os.listdir(inputdir) if '.tif' in f]
for f in files:
    print('processing file', f)
    img = imread(os.path.join(inputdir, f))
    dst = compute_indiv_sdt(img)
    tifffile.imwrite(os.path.join(outputdir, f), np.array(dst, dtype=np.float32))