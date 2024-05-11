import numpy as np
from .medfilt import medfilt2d as mf
    
def median_filter(ima, kernel_size=3):
    """
    Applies a median filter to a 2D image, ima.

    ima: np.array or np.ma.array of np.float64
    kernel_size: size of the kernel. 
    """
    if hasattr(ima, 'mask'):
        mask_ = ima.mask.astype(np.uint8)
        ima_ = ima.filled(fill_value=0)
    else:
        mask_ = np.zeros(ima.shape, dtype=np.int8)  
        ima_ = ima

    retval = mf(ima_, mask_, kernel_size, kernel_size, 1) 
                    
    return retval
