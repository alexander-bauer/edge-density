#!env/bin/python2
import argparse
import sys

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def main(args):
    for path in args.images:
        im = cv.imread(path, 0)

        dft = cv.dft(np.float32(im),flags = cv.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        mask = np.zeros((im.shape[0],im.shape[1],2),np.uint8)
        mask[:,:] = 1

        fshift = dft_shift * mask

        f_ishift = np.fft.ifftshift(fshift)

        im_back = cv.idft(f_ishift)
        im_back = cv.magnitude(im_back[:,:,0], im_back[:,:,1])

        cv.imshow("Input", im)
        cv.imshow("IFFT", im_back)
        cv.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+",
            help="images to check for occupancy")
    parser.add_argument("--size", type=int, default=800,
            help="maximum image size; larger images will be resized")
    parser.add_argument("--save", "-s", type=str, default=None,
            help="directory to save boxed results in if desired")

    sys.exit(main(parser.parse_args()))
