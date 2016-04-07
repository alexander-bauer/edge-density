#!env/bin/python2

from __future__ import print_function
import sys, os
import imutils
import argparse
import cv2 as cv

def main(args):
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    for path in args.images:
        im = cv.imread(path)
        # Resize the image. Smaller images will be processed faster.
        im = imutils.resize(im, width=min(args.size, im.shape[1]))

        # Detect humans using the pre-trained HOG model.
        rectangles, weights = hog.detectMultiScale(im,
                winStride=(2,2), padding=(8,8), scale=0.5)

        if args.save and not os.path.exists(args.save):
            os.makedirs(args.save)

        print(path, rectangles, weights)
        if args.draw or args.save:
            # Draw the rectangles on the image.
            for ((x, y, w, h), weight) in zip(rectangles, weights):
                # Color them green if certain, or red if uncertain.
                cv.rectangle(im,
                    (x, y), (x + w, y + h),
                    (0, int(255 * weight), 255 - int(255 * weight)), 2)
            
            if args.draw:
                # Show the image onscreen and wait for a keypress to dismiss.
                cv.imshow("Boxed Image", im)
                cv.waitKey(0)

            if args.save:
                boxed_path = os.path.join(args.save, os.path.basename(path))
                print(path, "->", boxed_path)
                cv.imwrite(boxed_path, im)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+",
            help="images to check for occupancy")
    parser.add_argument("--draw", "-d", action="store_true",
            help="draw bounding boxes around occupants")
    parser.add_argument("--size", type=int, default=800,
            help="maximum image size; larger images will be resized")
    parser.add_argument("--save", "-s", type=str, default=None,
            help="directory to save boxed results in if desired")

    sys.exit(main(parser.parse_args()))
