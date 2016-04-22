#!env/bin/python2

import imutils
import cv2 as cv

def check_occupancy(im):
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    # Detect humans using the pre-trained HOG model.
    rectangles, weights = hog.detectMultiScale(im,
        winStride=(2,2), padding=(8,8), scale=0.5)

    for ((x, y, w, h), weight) in zip(rectangles, weights):
        # Color them green if certain, or red if uncertain.
        cv.rectangle(im,
            (x, y), (x + w, y + h),
            (0, int(255 * weight), 255 - int(255 * weight)), 2)

    return im
