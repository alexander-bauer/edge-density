#!env/bin/python2
import argparse
import sys

import cv2 as cv
import numpy as np


def edges(im):
    # Convert to grayscale
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    return cv.Canny(im, 200, 400)

def corners(im, smoothing_sd=3):
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    smooth = cv.GaussianBlur(gray, (0, 0), smoothing_sd)
    dst = cv.cornerHarris(np.float32(smooth), 2, 3, 0.16)

    vis_points = cv.dilate(dst, None)
    im[vis_points > 0.01 * vis_points.max()] = [0,0,255]

    return im
