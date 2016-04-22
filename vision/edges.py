#!env/bin/python2
import argparse
import sys

import cv2 as cv
import numpy as np


def edges(im):
    # Convert to grayscale
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    dft = cv.dft(np.float32(im),flags = cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    mask = np.zeros((im.shape[0],im.shape[1],2),np.uint8)
    mask[:,:] = 1

    fshift = dft_shift * mask

    f_ishift = np.fft.ifftshift(fshift)

    im_back = cv.idft(f_ishift)
    im_back = cv.magnitude(im_back[:,:,0], im_back[:,:,1])

    return im_back
