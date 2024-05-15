from skimage.segmentation import find_boundaries, watershed
import numpy as np
from scipy.ndimage import label, maximum_filter, distance_transform_edt

def instance_to_semantic(labels: np.array) -> np.array: 
    """Convert instance segmentation to semantic segmentation with a class for background (0), cells (1) and boundaries (2)"""
    
    boundaries = find_boundaries(labels, mode = 'inner') > 0

    semantic_seg = np.zeros_like(labels) # background will stay at label 0
    semantic_seg[labels > 1] = 2 # foreground has label 2
    semantic_seg[boundaries] = 3 # boundaries have label 3 
    semantic_seg[labels == 1] = 1 # 1 is the label to be ignored

    return semantic_seg

def instance_to_semantic_per_slice(labels: np.array) -> np.array: 
    """Convert instance segmentation to semantic segmentation with a class for background (0), cells (1) and boundaries (2)"""
    
    semantic_seg_stack = np.zeros_like(labels)
    for i in range(labels.shape[0]):
        boundaries = find_boundaries(labels[i], mode = 'inner') > 0

        semantic_seg = np.zeros_like(labels[i]) # background will stay at label 0
        semantic_seg[labels[i] > 1] = 2 # foreground has label 2
        semantic_seg[boundaries] = 3 # boundaries have label 3 
        semantic_seg[labels[i] == 1] = 1 # 1 is the label to be ignored

        semantic_seg_stack[i] = semantic_seg

    return semantic_seg_stack

def compute_sdt(pred: np.ndarray, foreground_label = 2, background_label = 0, scale: int = 5):
    """Function to compute a signed distance transform."""

    # compute the distance transform inside and outside of the objects
    inner = np.zeros(pred.shape, dtype=np.float32)
    inner += distance_transform_edt(pred == foreground_label)
    outer = distance_transform_edt(pred == background_label)

    # create the signed distance transform
    distance = inner - outer

    # scale the distances so that they are between -1 and 1 (hint: np.tanh)
    distance = np.tanh(distance / scale)

    # be sure to return your solution as type 'float'
    return distance.astype(float)

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

def watershed_from_boundary_distance(
    boundary_distances: np.ndarray,
    inner_mask: np.ndarray,
    id_offset: float = 0,
    min_seed_distance: int = 20,
):
    """Function to compute a watershed from boundary distances."""

    seeds, n = find_local_maxima(boundary_distances, min_seed_distance)

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds != 0] += id_offset

    # calculate our segmentation
    segmentation = watershed(
        boundary_distances.max() - boundary_distances, seeds, mask=inner_mask
    )

    return seeds, segmentation

def get_inner_mask(pred, threshold):
    inner_mask = pred > threshold
    return inner_mask

def find_local_maxima(distance_transform, min_dist_between_points):
    # Use `maximum_filter` to perform a maximum filter convolution on the distance_transform
    max_filtered = maximum_filter(distance_transform, min_dist_between_points)
    maxima = max_filtered == distance_transform
    # Uniquely label the local maxima
    seeds, n = label(maxima)

    return seeds, n