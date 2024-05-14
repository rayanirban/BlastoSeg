from skimage.segmentation import find_boundaries
import numpy as np



def instance_to_semantic(labels: np.array) -> np.array: 
    """Convert instance segmentation to semantic segmentation with a class for background (0), cells (1) and boundaries (2)"""
    
    boundaries = find_boundaries(labels, mode = 'inner') > 0

    semantic_seg = np.zeros_like(labels)
    semantic_seg[labels > 0] = 1
    semantic_seg[boundaries] = 2

    return semantic_seg