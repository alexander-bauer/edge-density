#!env/bin/python2
import argparse
import sys

import cv2 as cv
import numpy as np

import vision.utils

def edges(im):
    # Convert to grayscale
    bw_im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    return cv.Canny(bw_im, 200, 400)

def corners(im, smoothing_sd=3):
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    smooth = cv.GaussianBlur(gray, (0, 0), smoothing_sd)
    dst = cv.cornerHarris(np.float32(smooth), 2, 3, 0.16)

    vis_points = cv.dilate(dst, None)
    im[vis_points > 0.01 * vis_points.max()] = [0,0,255]

    return im

def dense_corners(im, smoothing_sd=3, window_size=16, step_size=4):
    if type(window_size) != tuple:
        window_size = (window_size, window_size)

    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    smooth = cv.GaussianBlur(gray, (0, 0), smoothing_sd)
    corners = cv.cornerHarris(np.float32(smooth), 2, 3, 0.16)
    thresh_corners = np.zeros((im.shape[0], im.shape[1], 1), np.uint8)
    thresh_corners[corners > 0.01 * corners.max()] = 255

    num_corners = np.zeros((im.shape[0], im.shape[1], 1), np.uint8)

    for x, y, window in vision.utils.sliding_window(thresh_corners, window_size, step_size):
        num_corners[x, y] = cv.countNonZero(window)

    num_corners = num_corners * (255 / num_corners.max())
    ret,dst = cv.threshold(num_corners, 150, 255, cv.THRESH_BINARY)

    return dst

def box_dense_corners(im, **kwargs):
    corners = dense_corners(im, **kwargs)

    dilated_corners = cv.dilate(corners, None)
    im[dilated_corners > 0] = [0, 0, 255]
    return im

    contours, hierarchy = cv.findContours(corners, 3, 4)
    for contour in contours:
        print(im)
        x, y, w, h = cv.boundingRect(contour)
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, im.shape[0]), min(y + h, im.shape[1])
        cv.rectangle(im, (x0, y0), (x1, y1), (0,255,0), 2)


    return im

def dense_edges(im, size=(30, 30), threshold=50):
    edges_im = edges(im)

    sum_in_box = cv.boxFilter(edges_im, -1, size, normalize=True)
    dense_edges = cv.adaptiveThreshold(sum_in_box, 255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
            101, -10)

    return dense_edges

def contour_dense_edges(im, **kwargs):
    edges = dense_edges(im, **kwargs)

    _, contours, hierarchy = cv.findContours(edges,
            cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    final_contours = [cv.convexHull(contour) for contour in contours]

    return cv.drawContours(edges, final_contours, -1, (255, 255, 255), -1)

def contour_enclosed(im):
    # Convert to grayscale
    bw_im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    output = np.zeros(bw_im.shape, dtype=np.uint8)

    edges = cv.Canny(bw_im, 200, 400)
    # Find contours in the image, and build the hierarchy.
    _, contours, hierarchy = cv.findContours(edges,
            cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    # Draw only contours which are enclosed.
    for i in range(hierarchy.shape[1]):
        # If the element in the hierarchy has a parent, then it is enclosed, so
        # draw it.
        if hierarchy[:,i,3] >= 0:
            output = cv.drawContours(output, contours, i, 255, -1)

    return output

def sum_closed_edges(im, size = (31, 31)):
    """Combine dense_edges and contour_enclosed to give a high response in areas
    that edges are dense and contours are enclosed.
    """
    edges_im = edges(im)
    sum_in_box = cv.boxFilter(edges_im, -1, size, normalize=True)

    enclosed = cv.GaussianBlur(contour_enclosed(im), size, 5)
    weighted = cv.addWeighted(sum_in_box, 0.5, enclosed, 0.5, 0)

    thresh = cv.adaptiveThreshold(sum_in_box, 255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
            101, -10)

    return thresh
