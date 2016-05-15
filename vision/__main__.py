#!env/bin/python2

from __future__ import print_function
import sys, os
import argparse

import cv2 as cv
import numpy as np
import imutils

import vision.truth
import vision.stitch
import vision.check_occupancy
import vision.edges

def noop(im):
    return im

procedures = [
        noop,
        vision.check_occupancy.check_occupancy,
        vision.edges.edges,
        vision.edges.corners,
        vision.edges.dense_corners,
        vision.edges.box_dense_corners,
        vision.edges.dense_edges
]

def main(args):
    procedure = None
    for f in procedures:
        if f.__name__ == args.procedure:
            procedure = f
            break

    if args.save and not os.path.exists(args.save):
        os.makedirs(args.save)

    images = []
    for path in args.images:
        im = cv.imread(path)
        im = imutils.resize(im, width=min(args.size, im.shape[1]))
        images.append((path, im))

    if args.panorama:
        common_path = os.path.commonprefix([path for path, im in images])
        images = [common_path, vision.stitch.stitch([im for path, im in images])]

    for path, im in images:
        modified = procedure(im)

        # If we are comparing with the truth value, construct that layer.
        if args.truth:
            confusion, modified = vision.truth.compare(modified, path, args.truth_path)
            print("Confusion matrix:\n{}".format(confusion))

        # Convert the result to 3-color if it is 1-color.
        if len(modified.shape) < 3:
            result = np.zeros_like(im)
            if args.colorize in ['white', 'blue']:
                result[:,:,0] = modified
            if args.colorize in ['white', 'green']:
                result[:,:,1] = modified
            if args.colorize in ['white', 'red']:
                result[:,:,2] = modified
        else:
            result = modified

        # If the overlay argument is provided, sum the results with the original
        # to produce an overlay.
        if args.overlay:
            result = cv.addWeighted(im, 1, result, 1, 0.0)

        if args.draw:
            # Show the image onscreen and wait for a keypress to dismiss.
            cv.imshow("Original", im)
            cv.imshow("Result", result)
            cv.waitKey(0)

        if args.save:
            mod_path = os.path.join(args.save, os.path.basename(path))
            print(path, "->", mod_path)
            cv.imwrite(mod_path, result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("procedure", choices=[f.__name__ for f in procedures])
    parser.add_argument("images", nargs="+",
            help="images to check for occupancy")
    parser.add_argument("--draw", "-d", action="store_true",
            help="show images once generated")
    parser.add_argument("--panorama", "-p", action="store_true",
            help="stitch images together, rather than operating individually")
    parser.add_argument("--overlay", "-o", action="store_true",
            help="add modified image to original as an overlay")
    parser.add_argument("--colorize", action="store",
            choices=['white', 'red', 'blue', 'green'], default='white',
            help="add modified image to original as an overlay")
    parser.add_argument("--truth", "-T", action="store_true",
            help="show the ground truth overlay")
    parser.add_argument("--truth-path", "-tp", default="data/ground_truth")
    parser.add_argument("--size", type=int, default=800,
            help="maximum image size; larger images will be resized")
    parser.add_argument("--save", "-s", type=str, default=None,
            help="directory to save boxed results in if desired")

    sys.exit(main(parser.parse_args()))
