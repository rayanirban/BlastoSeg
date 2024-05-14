from skimage.segmentation import find_boundaries
import numpy as np

def normalize(image: np.ndarray, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    """
    Normalize an image with given mean and standard deviation.

    Parameters
    ----------
    image : np.ndarray
        Array containing single image or patch, 2D or 3D.
    mean : float, optional
        Mean value for normalization, by default 0.0.
    std : float, optional
        Standard deviation value for normalization, by default 1.0.

    Returns
    -------
    np.ndarray
        Normalized array.
    """
    return (image - mean) / std

def instance_to_semantic(labels: np.array) -> np.array: 
    """Convert instance segmentation to semantic segmentation with a class for background (0), cells (1) and boundaries (2)"""
    
    boundaries = find_boundaries(labels, mode = 'inner') > 0

    semantic_seg = np.zeros_like(labels)
    semantic_seg[labels > 0] = 1
    semantic_seg[boundaries] = 2

    return semantic_seg