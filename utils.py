from skimage.segmentation import find_boundaries
import numpy as np

def instance_to_semantic(labels: np.array) -> np.array: 
    """Convert instance segmentation to semantic segmentation with a class for background (0), cells (1) and boundaries (2)"""
    
    boundaries = find_boundaries(labels, mode = 'inner') > 0

    semantic_seg = np.zeros_like(labels) # background will stay at label 0
    semantic_seg[labels > 1] = 2 # foreground has label 2
    semantic_seg[boundaries] = 3 # boundaries have label 3 
    semantic_seg[labels == 1] = 1 # 1 is the label to be ignored

    return semantic_seg