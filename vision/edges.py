#!env/bin/python2
import argparse
import sys

import cv2 as cv
import numpy as np


def edges(im):
    # Convert to grayscale
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    return cv.Canny(im, 200, 400)

def dense_edges(im, rect_radius=10, sd=4):
    result = np.zeros((im.shape[0], im.shape[1], 1), np.uint8)
    im_edges = edges(im)
    blurred = cv.GaussianBlur(im_edges, (0, 0), sd)
    ret,thresholded = cv.threshold(blurred, 95, 255, cv.THRESH_BINARY)

    return thresholded
