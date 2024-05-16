from skimage.segmentation import find_boundaries
import numpy as np

def instance_to_semantic(labels: np.array) -> np.array: 
    """Convert instance segmentation to semantic segmentation with a class for background (0), cells (1) and boundaries (2)"""

    semantic_seg_stack = np.zeros_like(labels) # 118,256,256

    for i in range(0, labels.shape[0]):

        boundaries = find_boundaries(labels[i], mode = 'inner') > 0

        semantic_seg = np.zeros_like(labels[i])
        semantic_seg[labels[i]>1] = 2
        semantic_seg[boundaries] = 3
        semantic_seg[labels[i]==1] =1

        semantic_seg_stack[i] = semantic_seg

    return semantic_seg_stack