#!env/bin/python2

from __future__ import print_function
import sys, os
import argparse

import cv2 as cv
import imutils

import vision.check_occupancy
import vision.edges

procedures = [
        vision.check_occupancy.check_occupancy,
        vision.edges.edges,
        vision.edges.corners
]

def main(args):
    procedure = None
    for f in procedures:
        if f.__name__ == args.procedure:
            procedure = f
            break

    if args.save and not os.path.exists(args.save):
        os.makedirs(args.save)

    for path in args.images:
        im = cv.imread(path)
        # Resize the image. Smaller images will be processed faster.
        im = imutils.resize(im, width=min(args.size, im.shape[1]))

        modified = procedure(im)

        if args.draw:
            # Show the image onscreen and wait for a keypress to dismiss.
            cv.imshow("Original", im)
            cv.imshow("Modified", modified)
            cv.waitKey(0)

        if args.save:
            mod_path = os.path.join(args.save, os.path.basename(path))
            print(path, "->", mod_path)
            cv.imwrite(mod_path, modified)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("procedure", choices=[f.__name__ for f in procedures])
    parser.add_argument("images", nargs="+",
            help="images to check for occupancy")
    parser.add_argument("--draw", "-d", action="store_true",
            help="show images once generated")
    parser.add_argument("--size", type=int, default=800,
            help="maximum image size; larger images will be resized")
    parser.add_argument("--save", "-s", type=str, default=None,
            help="directory to save boxed results in if desired")

    sys.exit(main(parser.parse_args()))
