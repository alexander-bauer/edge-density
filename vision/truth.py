import os

import cv2 as cv
import numpy as np

def truth(path, truth_root):
    """Load the corresponding truth image."""
    # Get the common part between the image path and truth root.
    common = os.path.commonprefix([path, truth_root])
    # Find the part of the image path that is not common with the truth root.
    rel_impath = os.path.relpath(path, common)
    # Take that relative path from the truth root, and that is the truth image.
    truth_impath = os.path.join(truth_root, rel_impath)
    truth_im = cv.imread(truth_impath, cv.IMREAD_GRAYSCALE)
    if type(truth_im) == None.__class__:
        raise ValueError("no truth image found at '{}'".format(truth_impath))
    return truth_im

def compare(imregion, path, truth_root):
    """Compare the given image region with the associated ground truth.

    Green where the region matches the truth.
    Red where the truth is not covered by the region.
    Blue where the region is not atop the truth.
    """
    # Load the truth image for this region.
    truth_im = truth(path, truth_root)

    # Construct a 3-color image in BGR.
    result = np.zeros((imregion.shape[0], imregion.shape[1], 3),
            dtype=np.uint8)

    # Fill the blue channel, which is where region > truth
    false_positives = (imregion > truth_im)
    result[:,:,0][false_positives] = 255

    # Fill the green channel, which is where region == truth and truth > 0
    true_positives = (truth_im > 0) & (imregion == truth_im)
    result[:,:,1][true_positives] = 255

    # Fill the red channel, which is where the (truth > region)
    false_negatives = (truth_im > imregion)
    result[:,:,2][false_negatives] = 255

    return result
