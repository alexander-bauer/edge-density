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
    """Compare the given image region with the associated ground truth. Returns
    a tuple containing the confusion matrix and the comparison image.

    Green where the region covers the truth.
    Red where the truth is not covered by the region.
    Blue where the region is not atop the truth.
    """
    # Load the truth image for this region.
    truth_im = truth(path, truth_root)

    # Construct a 3-color image in BGR.
    result = np.zeros((imregion.shape[0], imregion.shape[1], 3),
            dtype=np.uint8)

    # True positives are the same in the ground truth and input image.
    true_positive = (truth_im > 0) & (imregion == truth_im)

    # False positives are hits in the input image and not the ground truth.
    false_positive = imregion > truth_im

    # False negatives are hits in the grouth truth and not the input image.
    false_negative = truth_im > imregion

    # True negatives are zero in both the ground truth and input image.
    true_negative = (truth_im == 0) & (imregion == 0)

    result[:,:,0][false_positive] = 255 # blue
    result[:,:,1][true_positive]  = 255 # green
    result[:,:,2][false_negative] = 255 # red

    pixels = float(truth_im.size)
    confusion = np.matrix([[true_positive.sum()/pixels, false_negative.sum()/pixels],
                           [false_positive.sum()/pixels, true_negative.sum()/pixels]])
    return confusion, result
